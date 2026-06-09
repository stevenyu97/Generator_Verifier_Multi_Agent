#!/usr/bin/env python3
"""CPU smoke test for ConvFinQA / FinQA wiring (no model required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_LLM_ROOT = Path(__file__).resolve().parent.parent.parent
_ROOT = _LLM_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.convfinqa.cfq_evaluator import answers_match, evaluated_line_ids, reward_filer_draft
from benchmarks.registry import get_dataset_config
from benchmarks.line_eval import set_active_dataset
from core.schemas import DraftLine, DraftReturn
from benchmarks.tax.tax_line_case import LineCaseConfig, classify_line_case_id, per_line_rz_rzp


def _make_case(data_root: Path, dataset: str) -> Path:
    split = data_root / dataset / "train"
    cases = sorted({p.parent for p in split.rglob("gold.json")})
    if cases:
        return cases[0]
    fixture = Path(__file__).resolve().parent / "fixtures" / f"{dataset}_sample"
    if (fixture / "gold.json").exists():
        print(f"[smoke:{dataset}] using fixture {fixture.name}")
        return fixture
    raise SystemExit(
        f"No converted cases under {split}; run download_data.py + convert.py"
    )


def _load_input(case_dir: Path) -> dict:
    with (case_dir / "input.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)


def run_smoke(dataset: str) -> None:
    set_active_dataset(dataset)
    ds = get_dataset_config(dataset)
    data_root = Path(__file__).resolve().parent / "data"  # benchmarks/convfinqa/data
    case_dir = _make_case(data_root, dataset)
    gold_path = case_dir / "gold.json"
    line_ids = list(evaluated_line_ids(gold_path))
    assert line_ids, f"No turns in {gold_path}"

    inp = _load_input(case_dir)
    print(f"[smoke:{dataset}] case={case_dir.name} turns={len(line_ids)}")

    with gold_path.open(encoding="utf-8") as f:
        gold = json.load(f)
    turn0 = gold["turns"][0]
    draft = DraftReturn(
        return_version=ds.return_version,
        lines=[
            DraftLine(
                form=ds.form_name,
                line=line_ids[0],
                description=str(turn0.get("question", ""))[:80],
                amount=float(turn0.get("answer", 0) or 0),
                rationale="smoke-test draft",
            )
        ],
        metadata={"case_id": inp.get("id", case_dir.name)},
    )
    score = reward_filer_draft(draft, gold_path)
    print(f"  reward_filer_draft (oracle draft): {score:.3f}")
    assert score > 0.0, "Oracle draft should match at least one turn"

    wrong = DraftReturn(
        return_version=ds.return_version,
        lines=[
            DraftLine(
                form=ds.form_name,
                line=line_ids[0],
                description="wrong",
                amount=999999.0,
                rationale="deliberately wrong",
            )
        ],
        metadata={},
    )
    assert reward_filer_draft(wrong, gold_path) < 1.0

    rz, rzp, touched = per_line_rz_rzp(
        draft,
        wrong,
        gold_path,
        line_cfg=LineCaseConfig(),
        draft_identical_abs_tol=1e-2,
    )
    cid = classify_line_case_id(
        rx=True,
        ry=True,
        ryp=False,
        rz=rz.get(line_ids[0], False),
        rzp=rzp.get(line_ids[0], False),
        touched=touched.get(line_ids[0], False),
    )
    print(f"  case taxonomy sample (correct draft + SAC + bad revise): case {cid}")
    assert answers_match(turn0.get("answer"), turn0)
    print(f"[smoke:{dataset}] OK")


def _count_cases(data_root: Path, dataset: str, split: str) -> int:
    split_dir = data_root / dataset / split
    if not split_dir.exists():
        return 0
    return sum(1 for _ in split_dir.rglob("gold.json"))


def main() -> None:
    data_root = Path(__file__).resolve().parent / "data"  # benchmarks/convfinqa/data
    for dataset in ("finqa", "convfinqa"):
        n_train = _count_cases(data_root, dataset, "train")
        n_test = _count_cases(data_root, dataset, "test")
        print(f"[data:{dataset}] train={n_train} test={n_test}")
        if n_train == 0:
            print(f"[skip] no converted train cases for {dataset}")
            continue
        run_smoke(dataset)
    print("[done] smoke tests passed")


if __name__ == "__main__":
    main()
