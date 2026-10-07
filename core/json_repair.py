"""Recovery of model-emitted JSON that a strict parse rejects.

Verifier and generator rollouts produce three recurring formatting artifacts
that invalidate otherwise well-formed payloads:

* typographic double quotes (``„``, ``“``) in place of ASCII delimiters,
* a stray backslash before a string-terminating quote,
* truncation, when generation hits the token cap mid-``evidence``.

The first two are pure delimiter damage and are safe to normalize anywhere. The
third yields a *partial* payload, so it is opt-in via ``allow_truncated`` and is
intended for offline analysis rather than for acting on during a rollout.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_STRAY_ESCAPE = re.compile(r'\\"(?=\s*[,}\]])')
_SMART_QUOTES = str.maketrans({c: '"' for c in "\u201c\u201d\u201e\u201f\u2033"})


def relax_escapes(text: str) -> str:
    """Drop a stray backslash before a string-terminating quote."""
    return _STRAY_ESCAPE.sub('"', text)


def normalize_quotes(text: str) -> str:
    """Map typographic double quotes onto ASCII delimiters."""
    return text.translate(_SMART_QUOTES)


def unescape_overquoted(text: str) -> str:
    """Undo model-emitted ``\\"`` key delimiters (literal backslash + quote)."""
    if '\\"' not in text:
        return text
    return text.replace('\\"', '"')


def bracket_state(frag: str) -> List[str]:
    """Open-bracket stack for ``frag``, ignoring brackets inside strings."""
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    return stack


def close_open_brackets(text: str) -> Optional[Dict[str, Any]]:
    """Rewind a truncated payload to the last element boundary and close it."""
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    in_str = False
    esc = False
    safe: Optional[int] = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            safe = i + 1
        elif ch == ",":
            safe = i
    if safe is None:
        return None
    frag = s[:safe].rstrip().rstrip(",")
    closing = "".join("}" if c == "{" else "]" for c in reversed(bracket_state(frag)))
    try:
        obj = json.loads(frag + closing)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def repair_json(
    text: str,
    *,
    allow_truncated: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse ``text`` under progressively more repair, reporting what was needed."""
    variants = [
        ([], text),
        (["stray_escape"], relax_escapes(text)),
        (["smart_quotes"], normalize_quotes(text)),
        (["smart_quotes", "stray_escape"], relax_escapes(normalize_quotes(text))),
        (["overquoted"], unescape_overquoted(text)),
        (
            ["smart_quotes", "overquoted"],
            unescape_overquoted(normalize_quotes(text)),
        ),
        (
            ["smart_quotes", "stray_escape", "overquoted"],
            unescape_overquoted(relax_escapes(normalize_quotes(text))),
        ),
    ]
    seen: set[str] = set()
    deduped: List[Tuple[List[str], str]] = []
    for labels, candidate in variants:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append((labels, candidate))
    variants = deduped
    # Delimiter repair is lossless, so exhaust it before rewinding a payload.
    for labels, candidate in variants:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed, labels
    if allow_truncated:
        for labels, candidate in variants:
            obj = close_open_brackets(candidate)
            if obj is not None:
                return obj, labels + ["truncated"]
    return None, []
