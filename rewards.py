"""Scalar rewards for on-policy GRPO (filer vs gold XML; verifier vs draft + gold)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from line_eval import line_strict_correctness_by_line_id
from schemas import DraftReturn, SafetyCase


def reward_filer_draft(draft: DraftReturn, expected_xml_path: Path) -> float:
    """Higher is better: strict by-line accuracy in [0, 1]."""
    from line_eval import reward_filer_draft as _dispatch

    return _dispatch(draft, expected_xml_path)


def _merge_verdicts(a: str, b: str) -> str:
    """Prefer the more severe / informative verdict when duplicates exist."""
    order: Dict[str, int] = {"missing": -1, "correct": 0, "plausible": 1, "suspicious": 2, "wrong": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def verdict_by_line_from_case(safety_case: SafetyCase) -> Dict[str, str]:
    """Merge duplicate line ids the same way as ``reward_verifier_safety_case``."""
    verdict_by_line: Dict[str, str] = {}
    for lf in safety_case.line_findings:
        lid = (lf.line or "").strip()
        if not lid:
            continue
        prev = verdict_by_line.get(lid, "missing")
        verdict_by_line[lid] = _merge_verdicts(prev, lf.verdict)
    return verdict_by_line


def line_verdict_binary_aligned(verdict: str, gold_ok: bool) -> bool:
    """Per-line 0/1-style *good* signal for SAC or full verifier (matches reward logic, binarized).

    ``missing`` / empty → False (usable as ``y'`` = 0 when verifier produced nothing for that line).
    """
    v = (verdict or "").strip()
    if v == "missing" or not v:
        return False
    if gold_ok:
        return v == "correct"
    if v in ("wrong", "suspicious"):
        return True
    if v == "plausible":
        return True
    if v == "correct":
        return False
    return False


def reward_verifier_safety_case(
    safety_case: SafetyCase,
    draft: DraftReturn,
    expected_xml_path: Path,
    *,
    w_fp: float = 1.0,
    w_tn: float = 0.35,
    w_fn: float = 1.0,
    w_tp: float = 1.0,
    w_plausible_on_error: float = 0.35,
    w_incomplete: float = 0.15,
) -> float:
    """Reward aligned with TaxCalcBench lines; penalizes false SAC on strictly correct lines.

    A line is treated as flagged (SAC-like) when the verdict is not ``correct``:
    ``plausible`` / ``suspicious`` / ``wrong`` all count as raising the line.
    """
    correctness = line_strict_correctness_by_line_id(draft, expected_xml_path)  # noqa: via line_eval
    verdict_by_line: Dict[str, str] = {}
    for lf in safety_case.line_findings:
        lid = (lf.line or "").strip()
        if not lid:
            continue
        prev = verdict_by_line.get(lid, "missing")
        verdict_by_line[lid] = _merge_verdicts(prev, lf.verdict)

    n = len(correctness)
    if n == 0:
        return 0.0

    raw = 0.0
    for line_id, gold_ok in correctness.items():
        v = verdict_by_line.get(line_id, "missing")

        if gold_ok:
            if v == "correct":
                raw += w_tn
            elif v in ("wrong", "suspicious", "plausible"):
                raw -= w_fp
            else:
                raw -= w_incomplete
        else:
            if v in ("wrong", "suspicious"):
                raw += w_tp
            elif v == "plausible":
                raw += w_plausible_on_error
            elif v == "correct":
                raw -= w_fn
            else:
                raw -= w_incomplete

    return raw / float(n)
