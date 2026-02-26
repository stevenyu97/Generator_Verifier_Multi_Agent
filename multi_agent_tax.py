from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Dict, Any, Tuple
import json
from pathlib import Path
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


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
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch and transformers are required. Install with: pip install torch transformers")
        # Prefer GPU; override with env TAX_DEVICE=cuda or TAX_DEVICE=cuda:0 if needed
        import os
        self.device = os.environ.get("TAX_DEVICE", "").strip().lower() or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TAX_DEVICE=cuda but CUDA is not available. Install PyTorch with CUDA or unset TAX_DEVICE for CPU.")
        print(f"Using device: {self.device}")
        # Resolve HF cache snapshot so tokenizer finds files; use slow tokenizer if sentencepiece missing
        _path = Path(model_path)
        if (_path / "snapshots").exists():
            _snapshots = list((_path / "snapshots").iterdir())
            if _snapshots:
                model_path = str(_snapshots[0])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)

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

        # Robustly extract JSON from the model output.
        text = assistant.strip()
        # 1) Strip <think> ... </think> blocks (reasoning). If no closing tag, remove from <think> to end.
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if "<think>" in text:
                text = text.split("<think>", 1)[0].strip() + (" " + text.split("</think>", 1)[-1].strip() if "</think>" in text else "")
            if "<think>" in text:
                idx = text.find("<think>")
                text = (text[:idx] + " " + text[idx:].replace("<think>", "", 1)).strip()
                text = text.split("<think>")[0].strip() if "<think>" in text else text
            # Remove everything from opening <think> to end (unclosed thinking block)
            if "<think>" in text:
                text = text[: text.find("<think>")].strip()

        # 2) If wrapped in markdown fences, keep only the fenced block.
        if "```" in text:
            first_tick = text.find("```")
            second_tick = text.find("```", first_tick + 3)
            if second_tick != -1:
                text = text[first_tick + 3 : second_tick].strip()

        # 3) Take the outermost {...} span as JSON.
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace : last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Model returned invalid JSON (syntax error): {e}. Raw excerpt: {assistant[:500]}..."
                ) from e

        raise ValueError(f"Model did not return JSON: {assistant[:500]}...")


def _extract_json_from_response(assistant: str) -> Optional[Dict[str, Any]]:
    """Strip <think> blocks and markdown fences, then return parsed JSON or None."""
    text = assistant.strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            text = text[: text.find("<think>")].strip()
    if "```" in text:
        first_tick = text.find("```")
        second_tick = text.find("```", first_tick + 3)
        if second_tick != -1:
            text = text[first_tick + 3 : second_tick].strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass
    return None


def _infer_approver_decision_from_text(raw: str) -> Dict[str, Any]:
    """Infer approver decision when model returns only <think> reasoning and no JSON."""
    lower = raw.lower()
    if any(
        phrase in lower
        for phrase in (
            "decision should be to approve",
            "the decision is to approve",
            "therefore, the decision is to approve",
            "so the decision is to approve",
        )
    ):
        return {
            "decision": "approve",
            "decision_confidence": 0.8,
            "comments": "Inferred from model reasoning (no JSON in response).",
            "required_changes": {"lines_to_recompute": []},
        }
    if "approve" in lower and "needs_revision" not in lower:
        return {
            "decision": "approve",
            "decision_confidence": 0.7,
            "comments": "Inferred from model reasoning (no JSON in response).",
            "required_changes": {"lines_to_recompute": []},
        }
    return {
        "decision": "needs_revision",
        "decision_confidence": 0.6,
        "comments": "Inferred from model reasoning (no JSON in response).",
        "required_changes": {"lines_to_recompute": []},
    }


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
- Output ONLY the JSON object. Do not use <think> tags or any other text. Reply with nothing but the JSON.
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
    user_text = json.dumps(payload, indent=2)
    raw_response = client.chat(APPROVER_SYSTEM_PROMPT, user_text, max_new_tokens=2048)
    raw = _extract_json_from_response(raw_response)
    if raw is None:
        raw = _infer_approver_decision_from_text(raw_response)

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
        data = json.load(f)
    # TaxCalcBench files wrap payload in "input"
    return data.get("input", data)


def run_showcase(
    output_path: Optional[Path] = None,
    case_dir: Optional[Path] = None,
    max_rounds: int = 1,
) -> Tuple[DraftReturn, List[Dict[str, Any]]]:
    """
    Run a small showcase: load one case, run one episode, print and optionally save output.
    """
    if output_path is None:
        output_path = Path(__file__).parent / "showcase_output.md"

    # Prefer dataset under LLM/Dataset if present
    if case_dir is None:
        alt = Path("/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-w2-minimal-wages-alaska")
        if alt.joinpath("input.json").exists():
            case_dir = alt
        else:
            case_dir = Path(__file__).parent / "dataset" / "tax_calc_bench" / "ty24" / "test_data" / "single-w2-minimal-wages-alaska"

    lines_out: List[str] = []
    def log(msg: str = "") -> None:
        lines_out.append(msg)
        print(msg)

    log("# Multi-Agent Tax Filing Showcase")
    log()
    log("## Input case")
    log(f"Case directory: `{case_dir}`")
    if case_dir.joinpath("input.json").exists():
        input_json = load_taxcalcbench_case(case_dir)
        log(f"Loaded input keys: {list(input_json.keys())[:10]}...")
        log()
    else:
        log("No input.json found; using minimal synthetic input.")
        input_json = {
            "return_header": {"tp_prior_year_agi": {"value": 0}},
            "return_data": {"residency_status": {"value": "us_citizen"}},
            "w2": [
                {
                    "employer_name": {"label": "Employer", "value": "Acme"},
                    "wages": {"label": "Box 1", "value": 45000},
                    "withholding": {"label": "Box 2", "value": 3500},
                }
            ],
        }
        log()

    log("## Running episode (Filer → Verifier → Approver)")
    log()

    client = QwenClient()
    final_draft, episode_history = run_episode(client, input_json, max_rounds=max_rounds)

    for i, h in enumerate(episode_history):
        log(f"### Round {i + 1}")
        log()
        log("**Draft return (excerpt)**")
        for line in h["draft"]["lines"][:8]:
            log(f"  - {line.get('form', '')} Line {line.get('line', '')}: {line.get('amount')} — {line.get('rationale', '')[:60]}...")
        if len(h["draft"]["lines"]) > 8:
            log(f"  ... and {len(h['draft']['lines']) - 8} more lines")
        log()
        log("**Safety case**")
        sc = h["safety_case"]
        log(f"  - Overall verdict: {sc.get('overall_verdict')} (confidence: {sc.get('overall_confidence')})")
        for lf in (sc.get("line_findings") or [])[:5]:
            log(f"  - {lf.get('form')} {lf.get('line')}: {lf.get('verdict')} — {lf.get('claim', '')[:50]}")
        log()
        log("**Approver decision**")
        d = h["decision"]
        log(f"  - Decision: {d.get('decision')}; comments: {d.get('comments', '')[:80]}")
        log()

    log("## Final draft (full)")
    log("```json")
    log(json.dumps({"lines": [asdict(l) for l in final_draft.lines]}, indent=2))
    log("```")
    log()
    log("## Full episode history (JSON)")
    log("```json")
    log(json.dumps(episode_history, indent=2, default=str))
    log("```")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines_out), encoding="utf-8")
    log()
    log(f"Output saved to: {output_path.absolute()}")

    return final_draft, episode_history



if __name__ == "__main__":
    import sys
    run_showcase(max_rounds=1)
