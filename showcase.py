"""Showcase runner: load case, run episode, save output."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from client import QwenClient
from environment import run_episode
from schemas import DraftReturn


def load_taxcalcbench_case(case_dir: Path) -> Dict[str, Any]:
    input_path = case_dir / "input.json"
    with input_path.open() as f:
        data = json.load(f)
    return data.get("input", data)


def run_showcase(
    output_path: Optional[Path] = None,
    case_dir: Optional[Path] = None,
    max_rounds: int = 1,
) -> Tuple[DraftReturn, List[Dict[str, Any]]]:
    """Run one episode, print progress, save report to output_path (default: showcase_output.md)."""
    if output_path is None:
        output_path = Path(__file__).parent / "showcase_output.md"

    if case_dir is None:
        alt = Path(
            "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-w2-minimal-wages-alaska"
        )
        if alt.joinpath("input.json").exists():
            case_dir = alt
        else:
            case_dir = (
                Path(__file__).parent
                / "dataset"
                / "tax_calc_bench"
                / "ty24"
                / "test_data"
                / "single-w2-minimal-wages-alaska"
            )

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
    final_draft, episode_history = run_episode(
        client, input_json, max_rounds=max_rounds
    )

    for i, h in enumerate(episode_history):
        log(f"### Round {i + 1}")
        log()
        log("**Draft return (excerpt)**")
        for line in h["draft"]["lines"][:8]:
            log(
                f"  - {line.get('form', '')} Line {line.get('line', '')}: {line.get('amount')} — {line.get('rationale', '')[:60]}..."
            )
        if len(h["draft"]["lines"]) > 8:
            log(f"  ... and {len(h['draft']['lines']) - 8} more lines")
        log()
        log("**Safety case**")
        sc = h["safety_case"]
        log(
            f"  - Overall verdict: {sc.get('overall_verdict')} (confidence: {sc.get('overall_confidence')})"
        )
        for lf in (sc.get("line_findings") or [])[:5]:
            log(
                f"  - {lf.get('form')} {lf.get('line')}: {lf.get('verdict')} — {lf.get('claim', '')[:50]}"
            )
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
