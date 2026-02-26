"""Filer, Verifier, and Approver agents."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional

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
            raw = client.generate_json(FILER_SYSTEM_PROMPT, payload, max_new_tokens=8192)
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


# Lines the Verifier must evaluate (same as TaxCalcBench). Sending only these shrinks payload and output.
VERIFIER_LINE_IDS = {"1a", "9", "10", "11", "12", "15", "16", "19", "24", "25d", "26", "27", "28", "29", "32", "33", "34", "35a", "37"}

def _trim_draft_for_verifier(draft: DraftReturn, max_rationale_len: int = 80) -> Dict[str, Any]:
    """Send only evaluation lines with short rationales so the model has room to output JSON."""
    trimmed = []
    for l in draft.lines:
        if (l.line or "").strip() not in VERIFIER_LINE_IDS:
            continue
        d = asdict(l)
        if len(d.get("rationale", "")) > max_rationale_len:
            d["rationale"] = d["rationale"][:max_rationale_len] + "..."
        trimmed.append(d)
    return {
        "return_version": draft.return_version,
        "lines": trimmed,
        "metadata": draft.metadata,
    }


def _trim_input_for_verifier(input_json: Dict[str, Any], max_return_data_keys: int = 50) -> Dict[str, Any]:
    """Shrink input for Verifier so the prompt leaves room for JSON output."""
    inp = input_json.get("input", input_json)
    if not isinstance(inp, dict):
        return input_json
    out = {"return_header": inp.get("return_header", {})}
    rd = inp.get("return_data", {})
    if isinstance(rd, dict) and len(rd) > max_return_data_keys:
        keys = list(rd.keys())[:max_return_data_keys]
        out["return_data"] = {k: rd[k] for k in keys}
    else:
        out["return_data"] = rd
    if "w2" in inp:
        out["w2"] = inp["w2"]
    return out


def verifier_agent(
    client: QwenClient, input_json: Dict[str, Any], draft: DraftReturn
) -> SafetyCase:
    print("  [Verifier] Evaluating draft (calling model)...")
    draft_return_trimmed = _trim_draft_for_verifier(draft)
    input_trimmed = _trim_input_for_verifier(input_json)
    payload = {
        "input": input_trimmed,
        "draft_return": draft_return_trimmed,
    }
    for attempt in range(2):
        try:
            if attempt > 0:
                print("  [Verifier] Retry after JSON error...")
            raw = client.generate_json(
                VERIFIER_SYSTEM_PROMPT, payload, max_new_tokens=8192
            )
            break
        except ValueError as e:
            if ("invalid JSON" in str(e) or "did not return JSON" in str(e)) and attempt == 0:
                continue
            raise

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
    return SafetyCase(
        claim_id=raw.get("claim_id", "safety-case-1"),
        overall_verdict=ov,
        overall_confidence=float(raw.get("overall_confidence", 0.0)),
        line_findings=line_findings,
    )


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
