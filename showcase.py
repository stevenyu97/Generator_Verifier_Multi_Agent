"""Showcase runner: load case, run episode, save output; dry-run option."""
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


def write_showcase_dry_run(output_path: Optional[Path] = None) -> None:
    """Write a static example showcase output (no model run). Use with --dry-run."""
    if output_path is None:
        output_path = Path(__file__).parent / "showcase_output.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = """# Multi-Agent Tax Filing Showcase

## Input case
Case directory: `/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-w2-minimal-wages-alaska`
Loaded input keys: return_header, return_data, w2, ...

## Running episode (Filer → Verifier → Approver)

### Round 1

**Draft return (excerpt)**
  - 1040 Line 1a: 45000 — Sum of W-2 box 1 wages from input.json
  - 1040 Line 9: 14600 — Standard deduction for single filer TY24
  - 1040 Line 11: 30400 — AGI minus standard deduction
  - 1040 Line 15: 3500 — Federal tax from tax table / computation
  - 1040 Line 19: 3500 — Total tax
  - 1040 Line 25d: 3500 — Withholding from W-2 box 2
  - 1040 Line 34: 0 — Overpayment (refund)
  ... and 0 more lines

**Safety case**
  - Overall verdict: uncertain (confidence: 0.72)
  - 1040 1a: plausible — WagesAmt matches sum of W-2 box 1
  - 1040 9: correct — Standard deduction for single TY24
  - 1040 11: plausible — AGI minus deduction calculation consistent
  - 1040 15: suspicious — Tax table lookup should be verified
  - 1040 25d: correct — Withholding matches input W-2 box 2

**Approver decision**
  - Decision: approve; comments: Verifier found no critical errors; minor uncertainty on tax computation accepted.

## Final draft (full)
```json
{
  "lines": [
    {"form": "1040", "line": "1a", "description": "Total amount from Form(s) W-2, box 1", "amount": 45000, "rationale": "Single W-2 box 1 value from input."},
    {"form": "1040", "line": "9", "description": "Standard deduction", "amount": 14600, "rationale": "Single filer standard deduction TY24."},
    {"form": "1040", "line": "11", "description": "Adjusted gross income minus standard deduction", "amount": 30400, "rationale": "45000 - 14600."},
    {"form": "1040", "line": "15", "description": "Taxable income tax", "amount": 3500, "rationale": "From tax table for single, 30400 taxable income."},
    {"form": "1040", "line": "25d", "description": "Federal withholding", "amount": 3500, "rationale": "W-2 box 2 from input."},
    {"form": "1040", "line": "34", "description": "Overpayment", "amount": 0, "rationale": "Tax equals withholding."}
  ]
}
```

## Full episode history (JSON)
*(One round: Filer → Verifier → Approver; decision: approve.)*

---
*This file was generated by the multi-agent tax showcase (dry run). Run with the Qwen3-4B model for live output.*
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"Dry-run showcase output saved to: {output_path.absolute()}")
