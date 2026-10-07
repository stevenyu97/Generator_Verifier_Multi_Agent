"""Verifier feedback format variants for SAC ablations."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _as_text(value: Any) -> str:
    """Flatten a SAC field that may be a string or a list of strings."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _line_finding(raw: Dict[str, Any], line_id: str) -> Dict[str, Any]:
    """Locate the finding for ``line_id``.

    The verifier emits the line under the key ``line``; ``line_id`` is accepted
    as an alias so hand-written fixtures keep working.
    """
    target = str(line_id).strip()
    for item in raw.get("line_findings") or []:
        if not isinstance(item, dict):
            continue
        for key in ("line", "line_id"):
            if str(item.get(key, "")).strip() == target:
                return item
    return {}


def format_verifier_feedback_for_generator(
    y_raw: Dict[str, Any],
    line_id: str,
    feedback_format: str,
) -> Dict[str, Any]:
    """Down-convert a structured SAC into the feedback channel under ablation.

    ``sac`` passes the structured payload through; ``critique`` keeps the same
    content as flat prose so the comparison isolates structure rather than
    information; ``binary`` keeps only the intervention bit.
    """
    fmt = (feedback_format or "sac").lower()
    if fmt == "binary":
        return {
            "target_line": line_id,
            "intervention": True,
            "message": "Verifier flagged this line for review.",
        }
    if fmt == "critique":
        lf = _line_finding(y_raw, line_id)
        parts: List[str] = []
        claim = _as_text(lf.get("claim"))
        if claim:
            parts.append(claim)
        argument = _as_text(lf.get("arguments") or lf.get("argument"))
        if argument:
            parts.append(argument)
        evidence = _as_text(lf.get("evidence"))
        if evidence:
            parts.append(evidence)
        if not parts:
            rationale = _as_text(lf.get("rationale")) or _as_text(
                y_raw.get("rationale")
            )
            if rationale:
                parts.append(rationale)
        critique = " ".join(parts).strip()
        if not critique:
            critique = json.dumps(y_raw, ensure_ascii=False)[:512]
        return {"target_line": line_id, "critique": critique[:512]}
    return y_raw
