"""Helpers shared by training and evaluation: trim payloads + parse verifier JSON."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from schemas import (
    DraftReturn,
    EvidenceItem,
    LineFinding,
    SafetyCase,
)

_VALID_VERDICTS = ("correct", "plausible", "suspicious", "wrong")
_VALID_OVERALL = ("accept", "uncertain", "reject")


def _trim_draft_for_verifier(draft: DraftReturn, max_rationale: int = 120) -> Dict[str, Any]:
    lines = []
    for l in draft.lines:
        d = asdict(l)
        r = d.get("rationale", "")
        if len(r) > max_rationale:
            d["rationale"] = r[:max_rationale] + "..."
        lines.append(d)
    return {"return_version": draft.return_version, "lines": lines, "metadata": draft.metadata}


def _trim_input_for_verifier(input_json: Dict[str, Any], max_keys: int = 60) -> Dict[str, Any]:
    """Cap ``return_data`` width so verifier prompts stay within context length."""
    inp = input_json.get("input", input_json)
    if not isinstance(inp, dict):
        return input_json
    out: Dict[str, Any] = {"return_header": inp.get("return_header", {})}
    rd = inp.get("return_data", {})
    if isinstance(rd, dict) and len(rd) > max_keys:
        out["return_data"] = {k: rd[k] for k in list(rd.keys())[:max_keys]}
    else:
        out["return_data"] = rd
    if "w2" in inp:
        out["w2"] = inp["w2"]
    return out


def _apply_verifier_numeric_checks(draft: DraftReturn, sc: SafetyCase) -> SafetyCase:
    """Downgrade an over-confident 'correct' on line 11 when the arithmetic doesn't check out."""
    amounts: Dict[tuple, float] = {}
    for dl in draft.lines:
        try:
            amt = float(dl.amount)
        except (TypeError, ValueError):
            amt = 0.0
        amounts[((dl.form or "1040").strip(), (dl.line or "").strip())] = amt
    a = lambda lid: amounts.get(("1040", lid), 0.0)  # noqa: E731
    if abs(a("11") - (a("9") - a("10"))) > 1e-6:
        for lf in sc.line_findings:
            if lf.form == "1040" and lf.line.strip() == "11" and lf.verdict == "correct":
                lf.verdict = "suspicious"
    return sc


def parse_verifier_raw_to_safety_case(
    raw: Dict[str, Any],
    draft: DraftReturn,
    *,
    sac_only: bool = True,
) -> SafetyCase:
    """Validate verifier JSON; optionally drop ``correct`` findings (SAC view)."""
    line_findings = []
    for lf in raw.get("line_findings", []):
        evidence = [
            EvidenceItem(
                type=e.get("type", ""),
                location=e.get("location", ""),
                value=e.get("value"),
            )
            for e in lf.get("evidence", [])
        ]
        v = lf.get("verdict", "plausible")
        if v not in _VALID_VERDICTS:
            v = "plausible"
        line_findings.append(
            LineFinding(
                form=lf.get("form", "1040"),
                line=lf.get("line", ""),
                claim=lf.get("claim", ""),
                verdict=v,
                confidence=float(lf.get("confidence", 0.0)),
                arguments=lf.get("arguments", []),
                evidence=evidence,
            )
        )
    ov = raw.get("overall_verdict", "uncertain")
    if ov not in _VALID_OVERALL:
        ov = "uncertain"
    sc = SafetyCase(
        claim_id=raw.get("claim_id", "safety-case-1"),
        overall_verdict=ov,
        overall_confidence=float(raw.get("overall_confidence", 0.0)),
        line_findings=line_findings,
    )
    sc = _apply_verifier_numeric_checks(draft, sc)
    if sac_only:
        sac = [lf for lf in sc.line_findings if lf.verdict != "correct"]
        sc = SafetyCase(
            claim_id=sc.claim_id,
            overall_verdict=sc.overall_verdict,
            overall_confidence=sc.overall_confidence,
            line_findings=sac,
        )
    return sc
