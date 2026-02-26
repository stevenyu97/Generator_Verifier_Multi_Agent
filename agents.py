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
    payload = {"input": input_json, "feedback": feedback}
    raw = client.generate_json(FILER_SYSTEM_PROMPT, payload)
    lines = [DraftLine(**line) for line in raw.get("lines", [])]
    return DraftReturn(
        return_version=raw.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=raw.get("metadata", {}),
    )


def verifier_agent(
    client: QwenClient, input_json: Dict[str, Any], draft: DraftReturn
) -> SafetyCase:
    payload = {
        "input": input_json,
        "draft_return": {
            "return_version": draft.return_version,
            "lines": [asdict(l) for l in draft.lines],
            "metadata": draft.metadata,
        },
    }
    raw = client.generate_json(VERIFIER_SYSTEM_PROMPT, payload)

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
