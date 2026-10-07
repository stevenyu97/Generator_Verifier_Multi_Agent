"""Offline SAC content quality (Q_SAC) over verifier outputs in step traces."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.client import extract_json_from_response
from core.json_repair import repair_json

_RULE_HINT = re.compile(r"\b(rule|irs|regulation|line|schedule|form)\b", re.I)


@dataclass
class SACQualityScore:
    line_id: str
    targets_line: bool
    has_claim: bool
    has_argument: bool
    has_evidence: bool
    rule_cited: bool
    score: float
    parse_status: str = "failed"
    line_found: bool = False
    repairs: tuple = ()


def _as_text(value: Any) -> str:
    """Flatten a SAC field that may be a string or a list of strings."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _line_finding(raw: Dict[str, Any], line_id: str) -> Optional[Dict[str, Any]]:
    """Locate the finding for ``line_id``.

    The verifier emits the line under the key ``line``; ``line_id`` is accepted
    as an alias so hand-written fixtures keep working.
    """
    target = str(line_id).strip()
    for lf in raw.get("line_findings") or []:
        if not isinstance(lf, dict):
            continue
        for key in ("line", "line_id"):
            if str(lf.get(key, "")).strip() == target:
                return lf
    return None


def score_sac_from_verifier_text(
    y_text: str,
    line_id: str,
    input_json: Optional[Dict[str, Any]] = None,
) -> SACQualityScore:
    """Heuristic Q_SAC in [0,1] without gold at inference."""
    raw: Dict[str, Any] = {}
    try:
        raw = extract_json_from_response(y_text) or {}
    except Exception:
        raw = {}
    parse_status = "strict" if raw else "failed"
    repairs: List[str] = []
    if not raw:
        repaired, repairs = repair_json(y_text, allow_truncated=True)
        raw = repaired or {}
        if raw:
            parse_status = "salvaged"

    finding = _line_finding(raw, line_id)
    lf = finding or {}
    claim = _as_text(lf.get("claim") or raw.get("claim"))
    argument = _as_text(
        lf.get("arguments")
        or lf.get("argument")
        or raw.get("arguments")
        or raw.get("argument")
    )
    evidence = _as_text(lf.get("evidence") or raw.get("evidence"))
    verdict = _as_text(lf.get("verdict")).lower()
    targets = line_id in y_text or bool(lf)
    rule_cited = bool(_RULE_HINT.search(claim + argument + evidence))

    grounded = False
    if input_json and evidence:
        blob = json.dumps(input_json, ensure_ascii=False).lower()
        tok = evidence[:40].lower()
        grounded = (tok in blob) if len(tok) > 8 else (len(evidence) > 10)

    has_claim = len(claim) > 5 or verdict not in ("", "correct")
    has_argument = len(argument) > 10
    has_evidence = len(evidence) > 10 or grounded

    parts = [
        0.25 if targets else 0.0,
        0.25 if has_claim else 0.0,
        0.25 if has_argument else 0.0,
        0.15 if has_evidence else 0.0,
        0.1 if rule_cited else 0.0,
    ]
    return SACQualityScore(
        line_id=line_id,
        targets_line=targets,
        has_claim=has_claim,
        has_argument=has_argument,
        has_evidence=has_evidence,
        rule_cited=rule_cited,
        score=min(1.0, sum(parts)),
        parse_status=parse_status,
        line_found=finding is not None,
        repairs=tuple(repairs),
    )


def aggregate_q_sac_from_traces(trace_root: Path) -> Dict[str, Any]:
    scores: List[float] = []
    n_sac = 0
    parse_counts = {"strict": 0, "salvaged": 0, "failed": 0}
    repair_counts: Dict[str, int] = {
        "truncated": 0,
        "stray_escape": 0,
        "smart_quotes": 0,
    }
    components = {
        "targets_line": 0,
        "line_found": 0,
        "has_claim": 0,
        "has_argument": 0,
        "has_evidence": 0,
        "rule_cited": 0,
    }
    for fp in sorted(trace_root.glob("step_*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for tr in data.get("transitions") or []:
            if str(tr.get("y_choice")) != "SAC":
                continue
            n_sac += 1
            lid = str(tr.get("line_id", ""))
            y_text = str(tr.get("y_text") or "")
            s = score_sac_from_verifier_text(y_text, lid)
            scores.append(s.score)
            parse_counts[s.parse_status] += 1
            for label in s.repairs:
                repair_counts[label] = repair_counts.get(label, 0) + 1
            for key in components:
                components[key] += int(getattr(s, key))
    denom = max(len(scores), 1)
    return {
        "n_sac_transitions": n_sac,
        "mean_q_sac": sum(scores) / denom,
        "parse_rates": {k: v / denom for k, v in parse_counts.items()},
        "repair_rates": {k: v / denom for k, v in repair_counts.items()},
        "component_rates": {k: v / denom for k, v in components.items()},
        "scores": scores,
    }
