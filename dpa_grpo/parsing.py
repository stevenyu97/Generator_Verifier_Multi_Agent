"""DPA-GRPO: parsing module."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.agents import parse_verifier_raw_to_safety_case
from core.client import extract_json_from_response
from core.schemas import DraftLine, DraftReturn



def build_chat_text(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    # Qwen3 exposes an ``enable_thinking`` switch on its chat template. When True
    # (the default) the template auto-inserts a <think> prelude, which the model
    # then fills with long chain-of-thought text that exhausts the token budget
    # and truncates the JSON output we actually want. We disable it here so the
    # model is expected to emit JSON directly.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

def try_parse_draft(text: str) -> Optional[DraftReturn]:
    data = extract_json_from_response(text.strip())
    if not data or "lines" not in data:
        return None
    try:
        lines = [DraftLine(**line) for line in data.get("lines", [])]
    except (TypeError, ValueError):
        return None
    return DraftReturn(
        return_version=data.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=data.get("metadata", {}),
    )

def try_parse_verifier_safety_case(text: str, draft: DraftReturn) -> Optional[Any]:
    data = extract_json_from_response(text.strip())
    if not data or "line_findings" not in data:
        return None
    try:
        return parse_verifier_raw_to_safety_case(data, draft, sac_only=False)
    except (TypeError, ValueError, KeyError):
        return None

def try_parse_line_draft(text: str, target_line: str) -> Optional[DraftLine]:
    """Parse one-line filer output JSON into a ``DraftLine``."""
    data = extract_json_from_response(text.strip())
    if not data:
        return None
    if isinstance(data.get("lines"), list):
        for item in data.get("lines", []):
            try:
                d = DraftLine(**item)
                if str(d.line).strip() == str(target_line):
                    return d
            except Exception:
                continue
    needed = {"form", "line", "description", "amount", "rationale"}
    if needed.issubset(set(data.keys())):
        try:
            d = DraftLine(
                form=str(data.get("form", "1040")),
                line=str(data.get("line", "")),
                description=str(data.get("description", "")),
                amount=float(data.get("amount", 0.0)),
                rationale=str(data.get("rationale", "")),
            )
            if str(d.line).strip() == str(target_line):
                return d
        except Exception:
            return None
    return None

def _parse_approver_decision(text: str) -> Optional[str]:
    """Return 'KEEP', 'REVISE', or None when the model output is unparseable."""
    raw = extract_json_from_response((text or "").strip()) or {}
    if isinstance(raw, dict):
        d = raw.get("decision")
        if isinstance(d, str):
            d_up = d.strip().upper()
            if d_up in ("KEEP", "REVISE"):
                return d_up
    return None

def _extract_line_verdict_from_raw(raw: Dict[str, Any], line_id: str) -> str:
    lfs = raw.get("line_findings", [])
    if isinstance(lfs, list):
        for lf in lfs:
            if str(lf.get("line", "")).strip() == str(line_id):
                return str(lf.get("verdict", "missing")).strip() or "missing"
    v = raw.get("verdict")
    if isinstance(v, str) and v:
        return v
    return "missing"
