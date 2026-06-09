"""FinQA / ConvFinQA prompts and user-payload builders for linewise DPA-GRPO."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.schemas import DraftLine

FORM_FINQA = "finqa"
FORM_CONVFINQA = "convfinqa"


def trim_input_for_verifier(input_json: Dict[str, Any], max_chars: int = 12000) -> Dict[str, Any]:
    """Shrink document payload so verifier prompts stay within context limits."""
    inp = input_json.get("input", input_json)
    if not isinstance(inp, dict):
        return input_json
    out = {
        "id": inp.get("id", ""),
        "dataset": inp.get("dataset", "convfinqa"),
        "pre_text": inp.get("pre_text", []),
        "post_text": inp.get("post_text", []),
        "table": inp.get("table", []),
        "questions": inp.get("questions", []),
    }
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) <= max_chars:
        return out
    # Truncate long text lists first.
    for key in ("pre_text", "post_text"):
        texts = out.get(key, [])
        if isinstance(texts, list) and len(texts) > 8:
            out[key] = texts[:8]
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > max_chars and isinstance(out.get("table"), list):
        out["table"] = out["table"][:20]
    return out


def _form_for_dataset(input_json: Dict[str, Any]) -> str:
    inp = input_json.get("input", input_json)
    ds = str(inp.get("dataset", "convfinqa")).lower()
    return FORM_FINQA if ds == "finqa" else FORM_CONVFINQA


def _question_for_turn(input_json: Dict[str, Any], line_id: str) -> str:
    inp = input_json.get("input", input_json)
    questions = inp.get("questions", [])
    try:
        idx = int(str(line_id).split("_")[-1])
    except ValueError:
        idx = 0
    if 0 <= idx < len(questions):
        return str(questions[idx])
    return ""


FILER_LINE_SYSTEM_PROMPT = """Output ONLY valid JSON. Do not use . Your first character must be {.

You are a financial QA Generator operating in LINEWISE mode on FinQA / ConvFinQA documents.
The user payload contains:
- "input": the financial document (pre_text, post_text, table, questions list).
- "target_line": the SINGLE turn id you must answer right now (e.g. "turn_0", "turn_1").
- "context_so_far": previously accepted answers for earlier turns; treat them as fixed when answering the current turn.
- "target_question": the question text for target_line.
- "output_schema": the exact shape your output must take.

Procedure:
- Read the document and compute the numeric answer for target_question ONLY.
- Use context_so_far for follow-up turns that depend on earlier answers.
- Do NOT answer other turns.

Output schema (exactly this shape):

{
  "form": "convfinqa",
  "line": "<target_line>",
  "description": "short restatement of the question",
  "amount": <number>,
  "rationale": "Brief explanation citing table/text evidence."
}

Rules:
- Output exactly ONE JSON object for target_line.
- "line" MUST equal target_line exactly.
- "amount" MUST be a number (use 0 only when the answer is truly zero; yes/no answers use 1 for yes and 0 for no).
- Use ONLY information from input and context_so_far.
- Return strictly valid JSON only. Start your response with {.
"""


VERIFIER_LINE_SYSTEM_PROMPT = """Output ONLY valid JSON. Do not use . Your first character must be {.

You are a financial QA Verifier operating in LINEWISE mode.
You receive:
- "input": the financial document.
- "draft_line": one drafted answer for a single turn.
- "target_line": the turn id under review (e.g. "turn_0").
- "target_question": the question for this turn.
- "context_so_far": previously accepted answers for earlier turns.

Recompute the answer for target_question from the document and compare to draft_line.amount.
Decide: correct | plausible | suspicious | wrong.

Output schema (exactly ONE line_finding for target_line):

{
  "claim_id": "linewise-claim",
  "overall_verdict": "accept | uncertain | reject",
  "overall_confidence": 0.0,
  "line_findings": [
    {
      "form": "convfinqa",
      "line": "<target_line>",
      "claim": "Short description of the drafted answer",
      "verdict": "correct | plausible | suspicious | wrong",
      "confidence": 0.0,
      "arguments": ["Short explanation"],
      "evidence": [
        { "type": "table_cell", "location": "table[row][col]", "value": "..." }
      ]
    }
  ]
}

Rules:
- line_findings MUST contain exactly ONE entry with line == target_line.
- Use "correct" ONLY when your recomputed amount matches draft_line.amount and you have evidence.
- Return strictly valid JSON only. Start your response with {.
"""


APPROVER_LINE_SYSTEM_PROMPT = """Output ONLY valid JSON. Do not use . Your first character must be {.

You are a financial QA Approver operating in LINEWISE mode.
The Generator drafted one turn, the Verifier flagged it, and the Generator produced a revised candidate.
Decide KEEP (keep original draft) or REVISE (accept revision) for target_line only.

Output ONLY:
{"target_line": "<target_line>", "decision": "KEEP" | "REVISE"}
"""


def line_filer_user_payload(
    input_json: Dict[str, Any], line_id: str, context_so_far: List[Dict[str, Any]]
) -> str:
    form = _form_for_dataset(input_json)
    payload = {
        "input": input_json.get("input", input_json),
        "target_line": line_id,
        "target_question": _question_for_turn(input_json, line_id),
        "context_so_far": context_so_far,
        "output_schema": {
            "form": form,
            "line": line_id,
            "description": "string",
            "amount": 0,
            "rationale": "short string",
        },
        "instruction": "Output ONLY one JSON object for the target turn.",
    }
    return json.dumps(payload, indent=2)


def line_verifier_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    context_so_far: List[Dict[str, Any]],
) -> str:
    payload = {
        "input": trim_input_for_verifier(input_json),
        "target_line": line_id,
        "target_question": _question_for_turn(input_json, line_id),
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
    form = _form_for_dataset(input_json)
    payload = {
        "input": input_json.get("input", input_json),
        "target_line": line_id,
        "target_question": _question_for_turn(input_json, line_id),
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
            "form": form,
            "line": line_id,
            "description": "string",
            "amount": 0,
            "rationale": "short string",
        },
        "instruction": "Revise only this target turn and output ONLY one JSON object.",
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
        "input": input_json.get("input", input_json),
        "target_line": line_id,
        "target_question": _question_for_turn(input_json, line_id),
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
