"""DPA-GRPO: checkpointing module."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpa_grpo.paths import checkpoint_root

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False
    PeftModel = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore



def _save_training_state(
    run_dir: Path,
    last_step: int,
    optimizer: AdamW,
    scaler: Optional[Any],
) -> None:
    payload: Dict[str, Any] = {
        "last_step": last_step,
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, run_dir / "training_state.pt")

def _assert_adapter_artifacts(save_dir: Path) -> None:
    has_weight = (save_dir / "adapter_model.safetensors").exists() or (
        save_dir / "adapter_model.bin"
    ).exists()
    if not has_weight:
        raise RuntimeError(
            f"Adapter weights were not saved in {save_dir} "
            "(expected adapter_model.safetensors or adapter_model.bin)"
        )
    if not (save_dir / "adapter_config.json").exists():
        raise RuntimeError(f"Missing adapter_config.json in {save_dir}")

def _save_lora_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    run_dir: Path,
    save_dir: Path,
    step_id: int,
    optimizer: AdamW,
    scaler: Optional[Any],
) -> None:
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    _assert_adapter_artifacts(save_dir)
    _save_training_state(run_dir, step_id, optimizer, scaler)
    if not (run_dir / "training_state.pt").exists():
        raise RuntimeError(f"Missing training_state.pt in {run_dir}")

def _resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if (p / "snapshots").exists():
        snaps = list((p / "snapshots").iterdir())
        if snaps:
            return str(snaps[0])
    return str(p)

def _set_active_adapter(model: torch.nn.Module, adapter_name: str) -> None:
    if _HAS_PEFT and PeftModel is not None and isinstance(model, PeftModel):
        model.set_adapter(adapter_name)

def _lora_target_modules(model: torch.nn.Module) -> List[str]:
    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found: set[str] = set()
    for n, _ in model.named_modules():
        for tail in names:
            if n.endswith(tail):
                found.add(tail)
    return sorted(found)
