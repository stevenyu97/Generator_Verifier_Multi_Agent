"""Scalar rewards for GRPO: filer draft accuracy and verifier safety-case score."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from evaluator import evaluate, line_strict_correctness_by_line_id
from schemas import DraftReturn, SafetyCase

_VERDICT_ORDER: Dict[str, int] = {
    "missing": -1, "correct": 0, "plausible": 1, "suspicious": 2, "wrong": 3,
}


def reward_filer_draft(draft: DraftReturn, expected_xml_path: Path) -> float:
    return float(evaluate(draft, expected_xml_path).correct_by_line_score)


def _merge_verdicts(a: str, b: str) -> str:
    return a if _VERDICT_ORDER.get(a, 0) >= _VERDICT_ORDER.get(b, 0) else b


def verdict_by_line_from_case(safety_case: SafetyCase) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lf in safety_case.line_findings:
        lid = (lf.line or "").strip()
        if not lid:
            continue
        out[lid] = _merge_verdicts(out.get(lid, "missing"), lf.verdict)
    return out


def line_verdict_binary_aligned(verdict: str, gold_ok: bool) -> bool:
    """1 when the verifier's per-line verdict matches gold_ok (used as binary verifier reward)."""
    v = (verdict or "").strip()
    if not v or v == "missing":
        return False
    if gold_ok:
        return v == "correct"
    return v in ("wrong", "suspicious", "plausible")


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
    """Per-line precision/recall-style reward over the 19 evaluated TaxCalcBench lines."""
    correctness = line_strict_correctness_by_line_id(draft, expected_xml_path)
    verdicts = verdict_by_line_from_case(safety_case)
    n = len(correctness)
    if n == 0:
        return 0.0
    raw = 0.0
    for lid, gold_ok in correctness.items():
        v = verdicts.get(lid, "missing")
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
