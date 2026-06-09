"""TaxCalcBench linewise user-payload builders (extracted for dataset_registry)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents import _trim_input_for_verifier
from schemas import DraftLine


def line_filer_user_payload(
    input_json: Dict[str, Any], line_id: str, context_so_far: List[Dict[str, Any]]
) -> str:
    payload = {
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context_so_far,
        "output_schema": {
            "form": "1040",
            "line": line_id,
            "description": "string",
            "amount": 0,
            "rationale": "short string",
        },
        "instruction": "Output ONLY one JSON object for the target line.",
    }
    return json.dumps(payload, indent=2)


def line_verifier_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    context_so_far: List[Dict[str, Any]],
) -> str:
    payload = {
        "input": _trim_input_for_verifier(input_json),
        "target_line": line_id,
        "context_so_far": context_so_far,
        "draft_line": {
            "form": line_obj.form,
            "line": line_obj.line,
            "description": line_obj.description,
            "amount": line_obj.amount,
            "rationale": line_obj.rationale,
        },
        "instruction": (
            "Output ONLY JSON with one line finding for target_line using verdict in "
            "{correct, plausible, suspicious, wrong}."
        ),
    }
    return json.dumps(payload, indent=2)


def line_revise_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    context_so_far: List[Dict[str, Any]],
) -> str:
    payload = {
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context_so_far,
        "draft_line": {
            "form": line_obj.form,
            "line": line_obj.line,
            "description": line_obj.description,
            "amount": line_obj.amount,
            "rationale": line_obj.rationale,
        },
        "verifier_feedback": verifier_raw,
        "output_schema": {
            "form": "1040",
            "line": line_id,
            "description": "string",
            "amount": 0,
            "rationale": "short string",
        },
        "instruction": "Revise only this target line and output ONLY one JSON object.",
    }
    return json.dumps(payload, indent=2)


def line_approver_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    revised_line: Optional[DraftLine],
    context_so_far: List[Dict[str, Any]],
) -> str:
    revised_block: Optional[Dict[str, Any]] = None
    if revised_line is not None:
        revised_block = {
            "form": revised_line.form,
            "line": revised_line.line,
            "description": revised_line.description,
            "amount": revised_line.amount,
            "rationale": revised_line.rationale,
        }
    payload = {
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context_so_far,
        "draft_line": {
            "form": line_obj.form,
            "line": line_obj.line,
            "description": line_obj.description,
            "amount": line_obj.amount,
            "rationale": line_obj.rationale,
        },
        "verifier_feedback": verifier_raw,
        "revised_line": revised_block,
        "instruction": (
            "Decide KEEP or REVISE for target_line and output ONLY the JSON "
            'object {"target_line": "...", "decision": "KEEP"|"REVISE"}.'
        ),
    }
    return json.dumps(payload, indent=2)
