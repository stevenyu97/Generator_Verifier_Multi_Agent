"""Data structures for the multi-agent tax filing pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal


@dataclass
class DraftLine:
    form: str  # e.g. "1040"
    line: str  # e.g. "1a"
    description: str
    amount: float
    rationale: str


@dataclass
class DraftReturn:
    return_version: str
    lines: List[DraftLine]
    metadata: Dict[str, Any]


@dataclass
class EvidenceItem:
    type: str  # e.g. "input_field", "calc_check"
    location: str  # path into input.json if applicable
    value: Any


@dataclass
class LineFinding:
    form: str
    line: str
    claim: str
    verdict: Literal["correct", "plausible", "suspicious", "wrong"]
    confidence: float
    arguments: List[str]
    evidence: List[EvidenceItem]


@dataclass
class SafetyCase:
    claim_id: str
    overall_verdict: Literal["accept", "uncertain", "reject"]
    overall_confidence: float
    line_findings: List[LineFinding]


@dataclass
class ApprovalDecision:
    decision: Literal["approve", "needs_revision"]
    decision_confidence: float
    comments: str
    required_changes: Dict[str, Any]  # e.g. {"lines_to_recompute": [{"form": "1040", "line": "9"}]}
