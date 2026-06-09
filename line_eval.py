"""Dataset-agnostic dispatch for per-line evaluation and rewards."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from schemas import DraftReturn

_active_dataset = "tax"


def set_active_dataset(name: str) -> None:
    global _active_dataset
    allowed = {"tax", "finqa", "convfinqa"}
    if name not in allowed:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(allowed)}")
    _active_dataset = name


def active_dataset() -> str:
    return _active_dataset


def _is_finqa_family() -> bool:
    return _active_dataset in {"finqa", "convfinqa"}


def evaluated_line_ids(gold_path: Path) -> Tuple[str, ...]:
    if _is_finqa_family():
        from convfinqa.cfq_evaluator import evaluated_line_ids as fn

        return fn(gold_path)
    from evaluator import evaluated_line_ids as fn

    return fn()


def draft_amount_for_line_id(draft: DraftReturn, line_id: str) -> float:
    if _is_finqa_family():
        from convfinqa.cfq_evaluator import draft_amount_for_line_id as fn

        return fn(draft, line_id)
    from evaluator import _draft_amount_for_line, LINES_TO_XPATH

    for line_desc in LINES_TO_XPATH:
        prefix = line_desc.split(":")[0].strip()
        lid = prefix.replace("Line ", "").strip() if prefix.startswith("Line ") else prefix
        if lid == line_id:
            return _draft_amount_for_line(draft, line_desc)
    return 0.0


def line_strict_correctness_by_line_id(
    draft: DraftReturn, gold_path: Path
) -> Dict[str, bool]:
    if _is_finqa_family():
        from convfinqa.cfq_evaluator import line_strict_correctness_by_line_id as fn

        return fn(draft, gold_path)
    from evaluator import line_strict_correctness_by_line_id as fn

    return fn(draft, gold_path)


def reward_filer_draft(draft: DraftReturn, gold_path: Path) -> float:
    if _is_finqa_family():
        from convfinqa.cfq_evaluator import reward_filer_draft as fn

        return fn(draft, gold_path)
    from evaluator import evaluate

    return float(evaluate(draft, gold_path).correct_by_line_score)
