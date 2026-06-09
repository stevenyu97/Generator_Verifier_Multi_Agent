"""Shared filesystem paths for DPA-GRPO artifacts."""
from __future__ import annotations

import os
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHECKPOINT_ROOT = Path(
    os.environ.get("DPA_GRPO_CHECKPOINT_ROOT", "/home/ubuntu/llm_artifacts/grpo_checkpoints")
)


def checkpoint_root() -> Path:
    return Path(os.environ.get("DPA_GRPO_CHECKPOINT_ROOT", str(DEFAULT_CHECKPOINT_ROOT)))
