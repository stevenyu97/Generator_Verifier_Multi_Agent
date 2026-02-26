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
    max_rounds: int = 3,
) -> Tuple[DraftReturn, List[Dict[str, Any]], Dict[str, int]]:
    """Run one episode, print progress, save report to output_path (default: showcase_output.md)."""
    if output_path is None:
        output_path = Path(__file__).parent / "showcase_output.md"

    if case_dir is None:
        alt = Path(
            "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/mfj-multiple-schedule-c-loss-multi-home-office"
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
                / "mfj-multiple-schedule-c-loss-multi-home-office"
            )

    lines_out: List[str] = []

    def log(msg: str = "") -> None:
        lines_out.append(msg)
        print(msg)

    log("# Multi-Agent Tax Filing Showcase")
    log()
    print("[Showcase] Loading input case...")
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

    print("[Showcase] Loading model and starting episode...")
    client = QwenClient()
    print("[Showcase] Running episode (Filer → Verifier → Approver)...")
    final_draft, episode_history, counts = run_episode(
        client, input_json, max_rounds=max_rounds
    )
    print("[Showcase] Episode finished. Building report...")

    log("## Agent agreement summary (one full tax case)")
    log()
    log("### Approver (round-level)")
    log(
        f"- **Agreed** (Approver approved the draft): {counts['agree_count']} time(s)"
    )
    log(
        f"- **Disagreed** (Approver requested revision): {counts['disagree_count']} time(s)"
    )
    log()
    # Verifier evaluates each line; summarize per-line verdicts from the last round
    if episode_history:
        sc = episode_history[-1].get("safety_case") or {}
        line_findings = sc.get("line_findings") or []
        by_verdict = {}
        for lf in line_findings:
            v = lf.get("verdict", "plausible")
            by_verdict[v] = by_verdict.get(v, 0) + 1
        total_findings = len(line_findings)
        log("### Verifier (per-line)")
        log("The Verifier should return one verdict per evaluation line (19 lines, same as TaxCalcBench). Counts from the final round:")
        if total_findings < 19:
            log(f"  _(Verifier returned only {total_findings} line finding(s); expected 19.)_")
        for v in ("correct", "plausible", "suspicious", "wrong"):
            n = by_verdict.get(v, 0)
            log(f"  - **{v}**: {n} line(s)")
        log()

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

    output_xml = Path(case_dir) / "output.xml"
    if output_xml.exists():
        try:
            print("[Showcase] Evaluating draft vs output.xml...")
            from evaluator import evaluate as evaluate_draft
            eval_result = evaluate_draft(final_draft, output_xml)
            log("## Evaluation (vs TaxCalcBench expected output.xml)")
            log()
            log(eval_result.report)
            log()
        except ImportError as e:
            log("## Evaluation skipped (install lxml to evaluate vs output.xml)")
            log(str(e))
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
    print(f"[Showcase] Writing report to {output_path}...")
    output_path.write_text("\n".join(lines_out), encoding="utf-8")
    log()
    log(f"Output saved to: {output_path.absolute()}")
    print("[Showcase] Done.")

    return final_draft, episode_history, counts
