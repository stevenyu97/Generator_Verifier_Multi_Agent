"""DPA-GRPO: data module."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dpa_grpo.config import _ds_cfg



def discover_case_dirs(root: Path, gold_filename: Optional[str] = None) -> List[Path]:
    """Directories under ``root`` that contain ``input.json`` and the gold artifact."""
    gold_name = gold_filename or _ds_cfg().gold_filename
    out: List[Path] = []
    for input_path in root.rglob("input.json"):
        d = input_path.parent
        if (d / gold_name).exists():
            out.append(d)
    return sorted(set(out))

def _select_case_for_step(
    cases: List[Path], step: int, cycle: bool, rng: random.Random
) -> Path:
    """Pick the training case for a given 1-indexed step.

    If ``cycle`` is True, iterate through ``cases`` deterministically in sorted
    order using ``(step - 1) % len(cases)`` so reruns are byte-identical. Else
    draw a random case from the pool (current default).
    """
    if not cases:
        raise ValueError("No cases provided to _select_case_for_step")
    if cycle:
        return cases[(max(int(step), 1) - 1) % len(cases)]
    return rng.choice(cases)

def split_train_test_cases(
    cases: List[Path], test_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if not cases:
        return [], []
    frac = max(0.0, min(0.95, float(test_fraction)))
    if frac <= 0.0 or len(cases) < 2:
        return list(cases), []
    n_test = int(round(len(cases) * frac))
    n_test = max(1, min(len(cases) - 1, n_test))
    ordered = sorted(cases, key=lambda p: str(p.resolve()))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    test_cases = sorted(ordered[:n_test], key=lambda p: str(p.resolve()))
    train_cases = sorted(ordered[n_test:], key=lambda p: str(p.resolve()))
    return train_cases, test_cases

def load_case_input(case_dir: Path) -> Dict[str, Any]:
    with (case_dir / "input.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)

def _draft_to_dict(d: Optional[DraftReturn]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {
        "return_version": d.return_version,
        "lines": [
            {
                "form": x.form,
                "line": x.line,
                "description": x.description,
                "amount": x.amount,
                "rationale": x.rationale,
            }
            for x in d.lines
        ],
        "metadata": d.metadata,
    }

def _write_step_trace(trace_dir: Path, gstep: int, payload: Dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    p = trace_dir / f"step_{gstep:05d}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
