"""JSON extraction from raw LLM completion text."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json_from_response(assistant: str) -> Optional[Dict[str, Any]]:
    """Strip <think> blocks and markdown fences, then return parsed JSON or None."""
    text = assistant.strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            text = text[: text.find("<think>")].strip()
    if "```" in text:
        a = text.find("```")
        b = text.find("```", a + 3)
        if b != -1:
            text = text[a + 3 : b].strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except json.JSONDecodeError:
            return None
    return None
