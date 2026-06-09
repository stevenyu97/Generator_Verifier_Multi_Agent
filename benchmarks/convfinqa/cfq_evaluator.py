"""FinQA / ConvFinQA evaluation against gold.json (tolerant numeric compare)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.schemas import DraftLine, DraftReturn

_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _load_gold(gold_path: Path) -> Dict[str, Any]:
    with gold_path.open(encoding="utf-8") as f:
        return json.load(f)


def _turns_from_gold(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = gold.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError("gold.json must contain a 'turns' list")
    return turns


def normalize_answer_text(raw: Any) -> Optional[Union[float, str]]:
    """Normalize FinQA-style answers to float or yes/no string."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"yes", "no"}:
        return low
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    val = float(m.group(0).replace(",", ""))
    if neg:
        val = -val
    return val


def _gold_targets(turn: Dict[str, Any]) -> Tuple[Optional[Union[float, str]], Optional[Union[float, str]]]:
    answer = turn.get("answer")
    answer_raw = turn.get("answer_raw", turn.get("answer_display"))
    norm_answer = normalize_answer_text(answer)
    norm_raw = normalize_answer_text(answer_raw) if answer_raw is not None else None
    return norm_answer, norm_raw


def answers_match(
    predicted: Any,
    gold_turn: Dict[str, Any],
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-2,
) -> bool:
    """Return True when predicted amount/text matches the gold turn."""
    pred_norm = normalize_answer_text(predicted)
    gold_norm, gold_raw_norm = _gold_targets(gold_turn)

    if pred_norm is None:
        return False

    # yes/no exact match
    if isinstance(pred_norm, str) or isinstance(gold_norm, str):
        pred_s = str(pred_norm).lower()
        gold_s = str(gold_norm or gold_raw_norm or "").lower()
        return pred_s == gold_s

    pred_f = float(pred_norm)
    candidates: List[float] = []
    for g in (gold_norm, gold_raw_norm):
        if isinstance(g, float):
            candidates.append(g)
        elif isinstance(g, str) and g in {"yes", "no"}:
            return False

    if not candidates:
        return False

    for gold_f in candidates:
        if abs(pred_f - gold_f) <= abs_tol:
            return True
        denom = max(abs(gold_f), 1.0)
        if abs(pred_f - gold_f) / denom <= rel_tol:
            return True
        # percentage vs fraction (e.g. 93.5 vs 0.935)
        if abs(pred_f / 100.0 - gold_f) <= abs_tol:
            return True
        if abs(pred_f - gold_f * 100.0) <= max(abs_tol, abs(gold_f) * 0.01):
            return True
    return False


def evaluated_line_ids(gold_path: Path) -> Tuple[str, ...]:
    gold = _load_gold(gold_path)
    return tuple(t["line_id"] for t in _turns_from_gold(gold))


def draft_amount_for_line_id(draft: DraftReturn, line_id: str) -> float:
    for d in draft.lines:
        if (d.line or "").strip() == line_id:
            try:
                return float(d.amount)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def line_strict_correctness_by_line_id(
    draft: DraftReturn, gold_path: Path
) -> Dict[str, bool]:
    gold = _load_gold(gold_path)
    turn_by_id = {t["line_id"]: t for t in _turns_from_gold(gold)}
    result: Dict[str, bool] = {}
    for line_id, turn in turn_by_id.items():
        amt = draft_amount_for_line_id(draft, line_id)
        result[line_id] = answers_match(amt, turn)
    return result


@dataclass
class EvaluationResult:
    strictly_correct_return: bool
    lenient_correct_return: bool
    correct_by_line_score: float
    lenient_correct_by_line_score: float
    report: str


def evaluate(draft: DraftReturn, gold_path: Path) -> EvaluationResult:
    correctness = line_strict_correctness_by_line_id(draft, gold_path)
    total = len(correctness)
    correct = sum(1 for ok in correctness.values() if ok)
    score = correct / total if total else 0.0
    lines_report = [
        f"{lid}: {'✓' if ok else '✗'}" for lid, ok in sorted(correctness.items())
    ]
    report = "\n".join(lines_report)
    report += f"\n\nCorrect (by turn): {score * 100:.2f}%"
    all_ok = correct == total and total > 0
    return EvaluationResult(
        strictly_correct_return=all_ok,
        lenient_correct_return=all_ok,
        correct_by_line_score=score,
        lenient_correct_by_line_score=score,
        report=report,
    )


def reward_filer_draft(draft: DraftReturn, gold_path: Path) -> float:
    return float(evaluate(draft, gold_path).correct_by_line_score)


def turn_question_for_line(gold_path: Path, line_id: str) -> str:
    gold = _load_gold(gold_path)
    for t in _turns_from_gold(gold):
        if t["line_id"] == line_id:
            return str(t.get("question", ""))
    return ""
