"""LLM client wrapper (Qwen3-4B) and JSON extraction helpers."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from core.device_pick import resolve_torch_device
except ImportError:
    resolve_torch_device = None  # type: ignore


def extract_json_from_response(assistant: str) -> Optional[Dict[str, Any]]:
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


def infer_approver_decision_from_text(raw: str) -> Dict[str, Any]:
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


class QwenClient:
    """
    Thin wrapper around a local Qwen3-4B model.
    Default path: /home/ubuntu/models/models--Qwen--Qwen3-4B
    """

    def __init__(self, model_path: str = "/home/ubuntu/models/models--Qwen--Qwen3-4B"):
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch and transformers are required. Install with: pip install torch transformers"
            )
        if resolve_torch_device is not None:
            self.device = resolve_torch_device()
        else:
            self.device = os.environ.get("TAX_DEVICE", "").strip() or (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "TAX_DEVICE requests CUDA but CUDA is not available. Install PyTorch with CUDA or set TAX_DEVICE=cpu."
            )
        print(f"Using device: {self.device}")
        _path = Path(model_path)
        if (_path / "snapshots").exists():
            _snapshots = list((_path / "snapshots").iterdir())
            if _snapshots:
                model_path = str(_snapshots[0])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)

    def chat(
        self, system_prompt: str, user_content: str, max_new_tokens: int = 1024
    ) -> str:
        """Run a single-turn chat and return the text response."""
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
        input_len = inputs["input_ids"].shape[1]
        # Stay under the model's max length to avoid the "exceeded predefined maximum length" warning
        model_max = getattr(self.model.config, "model_max_length", 40960)
        if input_len + max_new_tokens > model_max:
            max_new_tokens = max(256, model_max - input_len - 64)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def generate_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        max_new_tokens: int = 32768,
    ) -> Dict[str, Any]:
        user_text = json.dumps(user_payload, indent=2)
        assistant = self.chat(system_prompt, user_text, max_new_tokens=max_new_tokens)

        text = assistant.strip()
        # Remove closed think blocks
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # If model put JSON after </think>, use that (model sometimes ignores "no think" instruction)
        if "</think>" in text and "{" in text.split("</think>")[-1]:
            text = text.split("</think>")[-1].strip()
        # Remove unclosed think (model truncated or never closed)
        if "<think>" in text:
            text = text.split("<think>", 1)[0].strip()

        if "```" in text:
            first_tick = text.find("```")
            second_tick = text.find("```", first_tick + 3)
            if second_tick != -1:
                text = text[first_tick + 3 : second_tick].strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace : last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix trailing commas (common in model output)
                fixed = re.sub(r",\s*}", "}", json_str)
                fixed = re.sub(r",\s*]", "]", fixed)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Model returned invalid JSON (syntax error): {e}. Raw excerpt: {assistant[:500]}..."
                    ) from e

        # Fallback: take everything after last </think> and look for JSON (model often puts answer there)
        if "</think>" in assistant:
            after_think = assistant.split("</think>")[-1].strip()
            fb = after_think.find("{")
            lb = after_think.rfind("}")
            if fb != -1 and lb != -1 and lb > fb:
                try:
                    return json.loads(after_think[fb : lb + 1])
                except json.JSONDecodeError:
                    pass
        # Fallback: search raw response for any {...}
        first_brace = assistant.find("{")
        last_brace = assistant.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = assistant[first_brace : last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model did not return JSON: {assistant[:500]}...")
