"""DPA-GRPO: config module."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmarks.registry import get_dataset_config, gold_path_for_case
from benchmarks import line_eval
from core.schemas import DraftLine, DraftReturn
_ACTIVE_DATASET = "tax"


def configure_dataset(name: str) -> None:
    global _ACTIVE_DATASET
    _ACTIVE_DATASET = name
    line_eval.set_active_dataset(name)

def _ds_cfg():
    return get_dataset_config(_ACTIVE_DATASET)

def _gold_path(case_dir: Path) -> Path:
    return gold_path_for_case(case_dir, _ACTIVE_DATASET)

def _filer_line_prompt() -> str:
    return _ds_cfg().filer_line_system_prompt

def _approver_line_prompt() -> str:
    return _ds_cfg().approver_line_system_prompt

def _verifier_system_prompt_for(args: argparse.Namespace) -> str:
    """Linewise mode → single-line verifier prompt; otherwise full-return prompt."""
    ds = _ds_cfg()
    if bool(getattr(args, "linewise_rollout", False)):
        return ds.verifier_line_system_prompt
    return ds.verifier_system_prompt

def _line_filer_user_payload(
    input_json: Dict[str, Any], line_id: str, context_so_far: List[Dict[str, Any]]
) -> str:
    return _ds_cfg().line_filer_user_payload(input_json, line_id, context_so_far)

def _line_verifier_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_verifier_user_payload(
        input_json, line_obj, line_id, context_so_far
    )

def _line_revise_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_revise_user_payload(
        input_json, line_obj, line_id, verifier_raw, context_so_far
    )

def _line_approver_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    revised_line: Optional[DraftLine],
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_approver_user_payload(
        input_json, line_obj, line_id, verifier_raw, revised_line, context_so_far
    )
