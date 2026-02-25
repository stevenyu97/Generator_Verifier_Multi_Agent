from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Dict, Any, Tuple
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========= 1. Data structures =========


@dataclass
class DraftLine:
    form: str           # e.g. "1040"
    line: str           # e.g. "1a"
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
    type: str           # e.g. "input_field", "calc_check"
    location: str       # path into input.json if applicable
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


# ========= 2. Qwen client wrapper (local HF model) =========


class QwenClient:
    """
    Thin wrapper around a local Qwen3-4B model at:
    /home/ubuntu/models/models--Qwen--Qwen3-4B
    """

    def __init__(self, model_path: str = "/home/ubuntu/models/models--Qwen--Qwen3-4B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

    def chat(self, system_prompt: str, user_content: str, max_new_tokens: int = 1024) -> str:
        """
        Run a single-turn chat with system + user messages and return the text response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def generate_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        user_text = json.dumps(user_payload, indent=2)
        assistant = self.chat(system_prompt, user_text)
        first_brace = assistant.find("{")
        last_brace = assistant.rfind("}")
        if first_brace == -1 or last_brace == -1:
            raise ValueError(f"Model did not return JSON: {assistant}")
        json_str = assistant[first_brace:last_brace + 1]
        return json.loads(json_str)


# ========= 3. Agent role prompts =========


FILER_SYSTEM_PROMPT = """You are a tax calculation assistant for US Tax Year 2024 (federal only).
You receive a structured JSON object called "input" describing a taxpayer's situation.

Your job:
- Calculate key Form 1040 lines.
- Return ONLY a JSON object that matches this schema:

{
  "return_version": "ty24-v1",
  "lines": [
    {
      "form": "1040",
      "line": "1a",
      "description": "Total amount from Form(s) W-2, box 1",
      "amount": 47900,
      "rationale": "Explain how you got this number."
    }
  ],
  "metadata": {
    "filer_confidence": 0.0
  }
}

Rules:
- Use ONLY information from the input JSON.
- Do NOT fabricate forms or values that cannot be justified from input.
- Always include a brief human-readable rationale for each line.
- Return strictly valid JSON, with double quotes and no comments.
"""


VERIFIER_SYSTEM_PROMPT = """You are a tax safety Verifier.
You receive:
- "input": the original taxpayer input JSON.
- "draft_return": the draft Form 1040 JSON produced by a filer agent.

Your job:
- Critique the draft and produce a structured safety case as JSON with this schema:

{
  "claim_id": "global-claim-id",
  "overall_verdict": "accept | uncertain | reject",
  "overall_confidence": 0.0,
  "line_findings": [
    {
      "form": "1040",
      "line": "1a",
      "claim": "Short English description of what is being claimed",
      "verdict": "correct | plausible | suspicious | wrong",
      "confidence": 0.0,
      "arguments": ["English reasons referencing input fields and tax logic"],
      "evidence": [
        {
          "type": "input_field",
          "location": "path.into.input.json",
          "value": 123
        }
      ]
    }
  ]
}

Rules:
- Point to specific locations in the input JSON when possible (e.g. "w2[0].wages.value").
- Do NOT recompute a full new return; focus on evaluating the existing lines.
- Be conservative: if unsure, use "uncertain" or "suspicious" with appropriate confidence.
- Return strictly valid JSON only.
"""


APPROVER_SYSTEM_PROMPT = """You are the Approver of tax returns.
You receive:
- "input": taxpayer input JSON.
- "draft_return": current draft Form 1040 JSON.
- "safety_case": critique from the Verifier.

Your job:
- Decide whether to APPROVE the current draft or require a REVISION.
- Return JSON matching this schema:

{
  "decision": "approve" | "needs_revision",
  "decision_confidence": 0.0,
  "comments": "Short human-readable explanation",
  "required_changes": {
    "lines_to_recompute": [
      { "form": "1040", "line": "9" }
    ]
  }
}

Rules:
- If any important line is marked "wrong" or "suspicious" with moderate/high confidence, prefer "needs_revision".
- If all important lines are "correct" or "plausible" with high confidence, you may "approve".
- Return strictly valid JSON only.
"""


# ========= 4. Role functions =========


def filer_agent(client: QwenClient, input_json: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> DraftReturn:
    payload = {
        "input": input_json,
        "feedback": feedback,
    }
    raw = client.generate_json(FILER_SYSTEM_PROMPT, payload)

    lines = [DraftLine(**line) for line in raw.get("lines", [])]
    return DraftReturn(
        return_version=raw.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=raw.get("metadata", {}),
    )


def verifier_agent(client: QwenClient, input_json: Dict[str, Any], draft: DraftReturn) -> SafetyCase:
    payload = {
        "input": input_json,
        "draft_return": {
            "return_version": draft.return_version,
            "lines": [asdict(l) for l in draft.lines],
            "metadata": draft.metadata,
        },
    }
    raw = client.generate_json(VERIFIER_SYSTEM_PROMPT, payload)

    line_findings = []
    for lf in raw.get("line_findings", []):
        evidence = [EvidenceItem(**e) for e in lf.get("evidence", [])]
        line_findings.append(
            LineFinding(
                form=lf["form"],
                line=lf["line"],
                claim=lf["claim"],
                verdict=lf["verdict"],
                confidence=lf["confidence"],
                arguments=lf.get("arguments", []),
                evidence=evidence,
            )
        )
    return SafetyCase(
        claim_id=raw.get("claim_id", "safety-case-1"),
        overall_verdict=raw["overall_verdict"],
        overall_confidence=raw["overall_confidence"],
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
    raw = client.generate_json(APPROVER_SYSTEM_PROMPT, payload)

    return ApprovalDecision(
        decision=raw["decision"],
        decision_confidence=raw["decision_confidence"],
        comments=raw.get("comments", ""),
        required_changes=raw.get("required_changes", {"lines_to_recompute": []}),
    )


# ========= 5. Game loop for a single tax case =========


def run_episode(
    client: QwenClient,
    input_json: Dict[str, Any],
    max_rounds: int = 3,
) -> Tuple[DraftReturn, List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    draft = filer_agent(client, input_json, feedback=None)

    for round_idx in range(max_rounds):
        safety_case = verifier_agent(client, input_json, draft)
        decision = approver_agent(client, input_json, draft, safety_case)

        history.append(
            {
                "round": round_idx,
                "draft": {
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
                "decision": asdict(decision),
            }
        )

        if decision.decision == "approve":
            break

        if decision.decision == "needs_revision":
            feedback = {
                "safety_case": history[-1]["safety_case"],
                "decision": history[-1]["decision"],
            }
            draft = filer_agent(client, input_json, feedback=feedback)

    return draft, history


# ========= 6. Helpers to load a TaxCalcBench case =========


def load_taxcalcbench_case(case_dir: Path) -> Dict[str, Any]:
    input_path = case_dir / "input.json"
    with input_path.open() as f:
        return json.load(f)


if __name__ == "__main__":
    # Adjust this path to one of your test cases
    case_dir = Path(
        "/home/ubuntu/llm/dataset/tax_calc_bench/ty24/test_data/single-w2-minimal-wages-alaska"
    )
    input_json = load_taxcalcbench_case(case_dir)

    client = QwenClient()
    final_draft, episode_history = run_episode(client, input_json)

    print("Final draft lines:")
    print(json.dumps({"lines": [asdict(l) for l in final_draft.lines]}, indent=2))

