#!/usr/bin/env python3
"""Convert raw FinQA / ConvFinQA JSON into per-case dirs for DPA-GRPO."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from convfinqa.cfq_evaluator import normalize_answer_text

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT / "data"


def _safe_id(raw_id: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(raw_id).strip())
    return s[:180] or "case"


def _write_case(case_dir: Path, input_payload: Dict[str, Any], turns: List[Dict[str, Any]]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "input.json").write_text(
        json.dumps({"input": input_payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "gold.json").write_text(
        json.dumps({"turns": turns}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _turn_record(
    line_id: str,
    question: str,
    answer_raw: Any,
    program: Optional[str] = None,
) -> Dict[str, Any]:
    norm = normalize_answer_text(answer_raw)
    amount = float(norm) if isinstance(norm, float) else 0.0
    return {
        "line_id": line_id,
        "question": question,
        "answer": amount if isinstance(norm, float) else norm,
        "answer_raw": str(answer_raw),
        "program": program or "",
    }


def convert_finqa_split(raw_path: Path, out_split_dir: Path) -> int:
    with raw_path.open(encoding="utf-8") as f:
        records = json.load(f)
    count = 0
    for rec in records:
        qa = rec.get("qa", {})
        if not qa or "question" not in qa:
            continue
        case_id = _safe_id(rec.get("id", f"finqa_{count}"))
        case_dir = out_split_dir / case_id
        turns = [
            _turn_record(
                "turn_0",
                str(qa.get("question", "")),
                qa.get("exe_ans", qa.get("answer", "")),
                str(qa.get("program", "")),
            )
        ]
        input_payload = {
            "id": rec.get("id", case_id),
            "dataset": "finqa",
            "pre_text": rec.get("pre_text", []),
            "post_text": rec.get("post_text", []),
            "table": rec.get("table", []),
            "questions": [turns[0]["question"]],
        }
        _write_case(case_dir, input_payload, turns)
        count += 1
    return count


def _extract_conv_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Support official ConvFinQA (annotation), HF-cleaned (dialogue), and aliases."""
    if "dialogue" in rec and isinstance(rec["dialogue"], dict):
        return rec["dialogue"]
    if "annotation" in rec and isinstance(rec["annotation"], dict):
        return rec["annotation"]
    return {}


def _conv_questions(ann: Dict[str, Any]) -> List[str]:
    for key in (
        "conv_questions",
        "dialogue_break",
        "dialogue_break_ori",
        "questions",
    ):
        val = ann.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val]
    return []


def _conv_executed(ann: Dict[str, Any]) -> List[Any]:
    for key in (
        "executed_answers",
        "exe_ans_list",
        "conv_answers",
        "answer_list",
        "answers",
    ):
        val = ann.get(key)
        if isinstance(val, list) and val:
            return list(val)
    return []


def _conv_programs(ann: Dict[str, Any]) -> List[str]:
    for key in ("turn_program", "turn_program_ori", "programs"):
        val = ann.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val]
    return []


def convert_convfinqa_split(raw_path: Path, out_split_dir: Path) -> int:
    with raw_path.open(encoding="utf-8") as f:
        records = json.load(f)
    count = 0
    for rec in records:
        ann = _extract_conv_fields(rec)
        questions = _conv_questions(ann)
        executed = _conv_executed(ann)
        programs = _conv_programs(ann)
        if not questions:
            continue
        case_id = _safe_id(rec.get("id", f"convfinqa_{count}"))
        case_dir = out_split_dir / case_id
        turns: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            ans_raw = executed[i] if i < len(executed) else ""
            prog = programs[i] if i < len(programs) else ""
            turns.append(
                _turn_record(f"turn_{i}", str(q), ans_raw, str(prog) if prog else "")
            )
        input_payload = {
            "id": rec.get("id", case_id),
            "dataset": "convfinqa",
            "pre_text": rec.get("pre_text", []),
            "post_text": rec.get("post_text", []),
            "table": rec.get("table", {}),
            "questions": [t["question"] for t in turns],
        }
        _write_case(case_dir, input_payload, turns)
        count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["finqa", "convfinqa", "all"],
        default=["all"],
    )
    p.add_argument("--raw-dir", type=Path, default=RAW)
    p.add_argument("--out-dir", type=Path, default=DATA)
    cli = p.parse_args()
    want = set(cli.datasets)
    if "all" in want:
        want = {"finqa", "convfinqa"}

    if "finqa" in want:
        finqa_raw = cli.raw_dir / "finqa"
        for split, fname in [("train", "train.json"), ("test", "dev.json")]:
            src = finqa_raw / fname
            if not src.exists():
                raise SystemExit(f"Missing {src}; run download_data.py first")
            n = convert_finqa_split(src, cli.out_dir / "finqa" / split)
            print(f"[finqa/{split}] wrote {n} cases")

    if "convfinqa" in want:
        cfq_raw = cli.raw_dir / "convfinqa"
        # Official release unpacks JSON files at the zip root.
        for split, fname in [("train", "train.json"), ("test", "dev.json")]:
            candidates = [
                cfq_raw / fname,
                cfq_raw / "dataset" / fname,
                cfq_raw / "data" / fname,
            ]
            src = next((p for p in candidates if p.exists()), None)
            if src is None:
                raise SystemExit(
                    f"Missing ConvFinQA {fname}; run download_data.py first "
                    f"(looked in {candidates})"
                )
            n = convert_convfinqa_split(src, cli.out_dir / "convfinqa" / split)
            print(f"[convfinqa/{split}] wrote {n} cases from {src}")

    print(f"[done] normalized cases under {cli.out_dir}")


if __name__ == "__main__":
    main()
