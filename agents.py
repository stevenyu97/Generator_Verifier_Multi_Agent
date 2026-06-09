"""Filer, Verifier, and Approver agents."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

# Override with TAX_MAX_NEW_TOKENS if needed
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("TAX_MAX_NEW_TOKENS", "32768"))

from client import (
    QwenClient,
    extract_json_from_response,
    infer_approver_decision_from_text,
)
from prompts import (
    APPROVER_SYSTEM_PROMPT,
    FILER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from schemas import (
    ApprovalDecision,
    DraftLine,
    DraftReturn,
    EvidenceItem,
    LineFinding,
    SafetyCase,
)


def filer_agent(
    client: QwenClient,
    input_json: Dict[str, Any],
    feedback: Optional[Dict[str, Any]] = None,
) -> DraftReturn:
    if feedback is None:
        print("  [Filer] Generating initial draft (calling model)...")
    else:
        print("  [Filer] Revising draft (calling model)...")
    payload = {"input": input_json, "feedback": feedback}
    # Hard/complex cases need more tokens to output full draft (many lines)
    for attempt in range(2):
        try:
            if attempt > 0:
                print("  [Filer] Retry after JSON error...")
            raw = client.generate_json(FILER_SYSTEM_PROMPT, payload, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
            break
        except ValueError as e:
            if "invalid JSON" in str(e) and attempt == 0:
                continue  # Retry once on malformed/truncated JSON
            raise
    lines = [DraftLine(**line) for line in raw.get("lines", [])]
    return DraftReturn(
        return_version=raw.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=raw.get("metadata", {}),
    )


# Evaluation line ids (same as TaxCalcBench). Trimming Verifier payload to these keeps prompt under model max length.
_VERIFIER_LINE_IDS = {"1a", "9", "10", "11", "12", "15", "16", "19", "24", "25d", "26", "27", "28", "29", "32", "33", "34", "35a", "37"}


def _trim_draft_for_verifier(draft: DraftReturn, max_rationale: int = 120) -> Dict[str, Any]:
    """Send the full draft (all lines) with short rationales so the Verifier can see every line but the prompt stays within length limits."""
    lines = []
    for l in draft.lines:
        d = asdict(l)
        r = d.get("rationale", "")
        if len(r) > max_rationale:
            d["rationale"] = r[:max_rationale] + "..."
        lines.append(d)
    return {"return_version": draft.return_version, "lines": lines, "metadata": draft.metadata}


def _trim_input_for_verifier(input_json: Dict[str, Any], max_keys: int = 60) -> Dict[str, Any]:
    """Shrink input so Verifier prompt stays within model context limit."""
    inp = input_json.get("input", input_json)
    if not isinstance(inp, dict):
        return input_json
    # FinQA / ConvFinQA documents use pre_text/post_text/table, not tax return_data.
    if "return_data" not in inp and ("pre_text" in inp or "table" in inp):
        try:
            from convfinqa.cfq_prompts import trim_input_for_verifier as _cfq_trim

            return _cfq_trim(input_json)
        except ImportError:
            return inp
    out = {"return_header": inp.get("return_header", {})}
    rd = inp.get("return_data", {})
    if isinstance(rd, dict) and len(rd) > max_keys:
        out["return_data"] = {k: rd[k] for k in list(rd.keys())[:max_keys]}
    else:
        out["return_data"] = rd
    if "w2" in inp:
        out["w2"] = inp["w2"]
    return out


def verifier_agent(
    client: QwenClient, input_json: Dict[str, Any], draft: DraftReturn
) -> SafetyCase:
    print("  [Verifier] Evaluating draft (calling model)...")
    payload = {
        "input": _trim_input_for_verifier(input_json),
        "draft_return": _trim_draft_for_verifier(draft),
    }
    for attempt in range(2):
        try:
            if attempt > 0:
                print("  [Verifier] Retry after JSON error...")
            raw = client.generate_json(
                VERIFIER_SYSTEM_PROMPT, payload, max_new_tokens=DEFAULT_MAX_NEW_TOKENS
            )
            break
        except ValueError as e:
            if ("invalid JSON" in str(e) or "did not return JSON" in str(e)) and attempt == 0:
                continue
            raise

    return parse_verifier_raw_to_safety_case(raw, draft, sac_only=True)


def _apply_verifier_numeric_checks(draft: DraftReturn, safety_case: SafetyCase) -> SafetyCase:
    """Lightweight numeric sanity checks that can upgrade overly-optimistic Verifier verdicts.

    These checks are deterministic and operate only on the draft itself, not on ground-truth XML.
    If a line's amount is arithmetically inconsistent with related lines but the Verifier marked
    it as 'correct', we downgrade it to 'suspicious' so downstream analysis sees a sac for it.
    """
    # Index draft lines by (form, line) for quick lookup
    amounts: Dict[tuple, float] = {}
    for dl in draft.lines:
        try:
            amt = float(dl.amount)
        except (TypeError, ValueError):
            amt = 0.0
        key = ((dl.form or "1040").strip(), (dl.line or "").strip())
        amounts[key] = amt

    # Helper to get 1040 line amount with default 0.0
    def _amt(line_id: str) -> float:
        return amounts.get(("1040", line_id), 0.0)

    # Tax-only internal-consistency rule: Line 11 ≈ Line 9 − Line 10
    if ("1040", "11") not in amounts:
        return safety_case
    target = _amt("11")
    lhs = _amt("9") - _amt("10")
    if abs(target - lhs) > 1e-6:
        for lf in safety_case.line_findings:
            if lf.form == "1040" and lf.line.strip() == "11" and lf.verdict == "correct":
                lf.verdict = "suspicious"

    return safety_case


def parse_verifier_raw_to_safety_case(
    raw: Dict[str, Any],
    draft: DraftReturn,
    *,
    sac_only: bool = True,
) -> SafetyCase:
    """Parse verifier JSON into a ``SafetyCase``.

    Numeric checks run on the full finding list first. If ``sac_only`` (default),
    findings with verdict ``correct`` are then removed for SAC-style payloads.
    Use ``sac_only=False`` for training rewards over all 19 evaluated lines.
    """
    _valid_verdicts = ("correct", "plausible", "suspicious", "wrong")
    _valid_overall = ("accept", "uncertain", "reject")
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
        if v not in _valid_verdicts:
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
    if ov not in _valid_overall:
        ov = "uncertain"

    safety_case = SafetyCase(
        claim_id=raw.get("claim_id", "safety-case-1"),
        overall_verdict=ov,
        overall_confidence=float(raw.get("overall_confidence", 0.0)),
        line_findings=line_findings,
    )
    safety_case = _apply_verifier_numeric_checks(draft, safety_case)
    if sac_only:
        sac = [lf for lf in safety_case.line_findings if lf.verdict != "correct"]
        safety_case = SafetyCase(
            claim_id=safety_case.claim_id,
            overall_verdict=safety_case.overall_verdict,
            overall_confidence=safety_case.overall_confidence,
            line_findings=sac,
        )
    return safety_case


def approver_agent(
    client: QwenClient,
    input_json: Dict[str, Any],
    draft: DraftReturn,
    safety_case: SafetyCase,
) -> ApprovalDecision:
    payload = {
        "input": input_json,
        "draft_return": {
            "return_version": draft.return_version,
            "lines": [asdict(l) for l in draft.lines],
            "metadata": draft.metadata,
        },
        "safety_case": {
            "claim_id": safety_case.claim_id,
            "overall_verdict": safety_case.overall_verdict,
            "overall_confidence": safety_case.overall_confidence,
            "line_findings": [asdict(lf) for lf in safety_case.line_findings],
        },
    }
    user_text = json.dumps(payload, indent=2)
    print("  [Approver] Deciding approve / needs_revision (calling model)...")
    raw_response = client.chat(APPROVER_SYSTEM_PROMPT, user_text, max_new_tokens=2048)
    raw = extract_json_from_response(raw_response)
    if raw is None:
        raw = infer_approver_decision_from_text(raw_response)

    return ApprovalDecision(
        decision=raw["decision"],
        decision_confidence=raw["decision_confidence"],
        comments=raw.get("comments", ""),
        required_changes=raw.get("required_changes", {"lines_to_recompute": []}),
    )


def verify_and_approve_line(
    client: QwenClient,
    input_json: Dict[str, Any],
    draft: DraftReturn,
    line_id: str,
) -> Tuple[LineFinding, ApprovalDecision]:
    """Run Verifier and Approver but return the finding and decision for a single line."""
    safety_case = verifier_agent(client, input_json, draft)

    # Find the verifier finding corresponding to the requested line_id
    line_finding = next(
        (lf for lf in safety_case.line_findings if lf.line == line_id),
        None,
    )
    if line_finding is None:
        raise ValueError(f"No verifier finding for line {line_id!r}")

    # Construct a minimal safety_case focused on this line for the Approver
    per_line_safety_case = SafetyCase(
        claim_id=safety_case.claim_id,
        overall_verdict=safety_case.overall_verdict,
        overall_confidence=safety_case.overall_confidence,
        line_findings=[line_finding],
    )

    decision = approver_agent(client, input_json, draft, per_line_safety_case)
    return line_finding, decision
