import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from environment import run_episode
from evaluator import (
    LINES_TO_XPATH,
    evaluate as evaluate_draft_fn,
    _draft_amount_for_line,
    _parse_xml_value,
)
from gpt5_client import Gpt5Client
from schemas import DraftLine, DraftReturn


def _safe_amount(raw: Any) -> float:
    """Coerce model output (e.g. '' or non-numeric) to float for draft line amount."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def load_taxcalcbench_case(case_dir: Path) -> Dict[str, Any]:
    input_path = case_dir / "input.json"
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)


def run_gpt5_multiagent_showcase(
    case_dir: Optional[Path] = None,
    max_rounds: int = 3,
    output_path: Optional[Path] = None,
) -> Tuple[DraftReturn, List[Dict[str, Any]]]:
    """
    Baseline: run the full Filer → Verifier → Approver loop using GPT-5
    (same model as gpt5_filer.py). Logs full interaction and writes a report
    like showcase_output.md to output_path (default: gpt5_showcase_output.md).
    """
    if case_dir is None:
        case_dir = Path(
            "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-retirement-1099r-alaska-dividend"
            #"/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/mfj-multiple-schedule-c-loss-multi-home-office"
        )
    if output_path is None:
        output_path = Path(__file__).parent / f"gpt5_showcase_output_{case_dir.name}.md"

    input_path = case_dir / "input.json"
    output_xml_path = case_dir / "output.xml"
    if not input_path.exists():
        raise FileNotFoundError(f"input.json not found in {case_dir}")
    if not output_xml_path.exists():
        raise FileNotFoundError(f"output.xml not found in {case_dir}")

    lines_out: List[str] = []

    def log(msg: str = "") -> None:
        lines_out.append(msg)
        print(msg)

    log("# GPT-5 Multi-Agent Tax Filing Showcase")
    log()
    print("[GPT5 Multi-Agent] Loading input case...")
    log("## Input case")
    log(f"Case directory: `{case_dir}`")
    input_json = load_taxcalcbench_case(case_dir)
    log(f"Loaded input keys: {list(input_json.keys())[:10]}...")
    log()

    log("## Running episode (Filer → Verifier → Approver)")
    log()

    print("[GPT5 Multi-Agent] Instantiating GPT-5 client...")
    client = Gpt5Client()
    print("[GPT5 Multi-Agent] Running episode (Filer → Verifier → Approver)...")
    final_draft, episode_history, _ = run_episode(
        client, input_json, max_rounds=max_rounds
    )
    print("[GPT5 Multi-Agent] Episode finished. Building report...")

    log("## Agent agreement summary (one full tax case)")
    log()
    if episode_history:
        last_round = episode_history[-1]
        sc = last_round.get("safety_case") or {}
        line_findings = sc.get("line_findings") or []
        by_verdict = {}
        for lf in line_findings:
            v = lf.get("verdict", "plausible")
            by_verdict[v] = by_verdict.get(v, 0) + 1
        total_findings = len(line_findings)
        log("### Verifier (per-line)")
        log("Counts from the final round:")
        if total_findings < 19:
            log(f"  _(Verifier returned only {total_findings} line finding(s); expected 19.)_")
        for v in ("correct", "plausible", "suspicious", "wrong"):
            n = by_verdict.get(v, 0)
            log(f"  - **{v}**: {n} line(s)")
        log()

        last_decision = last_round.get("decision") or {}
        decision_label = last_decision.get("decision")
        recompute_lines = {
            (c.get("form"), c.get("line"))
            for c in (
                (last_decision.get("required_changes") or {}).get(
                    "lines_to_recompute", []
                )
            )
        }

        log("### Per-line trends (Verifier + Approver, final round)")
        for lf in line_findings:
            form = lf.get("form", "")
            line_no = lf.get("line", "")
            verdict = lf.get("verdict", "plausible")
            key = (form, line_no)
            if decision_label == "approve":
                approver_status = "approved"
            elif key in recompute_lines and decision_label == "needs_revision":
                approver_status = "needs_revision"
            else:
                approver_status = "not_flagged"
            log(
                f"- {form} Line {line_no}: Verifier **{verdict}**, Approver **{approver_status}**"
            )
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

        sc = h["safety_case"]
        d = h["decision"]
        decision_label = d.get("decision")
        recompute_lines_round = {
            (c.get("form"), c.get("line"))
            for c in (
                (d.get("required_changes") or {}).get(
                    "lines_to_recompute", []
                )
            )
        }

        log("**Safety case (per-line Verifier verdicts)**")
        log(
            f"  - Overall verdict: {sc.get('overall_verdict')} (confidence: {sc.get('overall_confidence')})"
        )
        for lf in (sc.get("line_findings") or []):
            log(
                f"  - {lf.get('form')} {lf.get('line')}: {lf.get('verdict')} — {lf.get('claim', '')[:50]}"
            )
        log()

        log("**Per-line interaction (Verifier + Approver for this round)**")
        for lf in (sc.get("line_findings") or []):
            form = lf.get("form", "")
            line_no = lf.get("line", "")
            verdict = lf.get("verdict", "plausible")
            key = (form, line_no)
            if decision_label == "approve":
                approver_status = "approved"
            elif key in recompute_lines_round and decision_label == "needs_revision":
                approver_status = "needs_revision"
            else:
                approver_status = "not_flagged"
            log(
                f"  - {form} Line {line_no}: Verifier **{verdict}**, Approver **{approver_status}**"
            )
        log()

        log("**Approver decision (round summary)**")
        log(f"  - Decision: {decision_label}; comments: {d.get('comments', '')[:80]}")
        log()

    try:
        print("[GPT5 Multi-Agent] Evaluating draft vs output.xml...")
        correct_before_pct = None
        if episode_history:
            first_round_draft = episode_history[0].get("draft") or {}
            initial_draft = DraftReturn(
                return_version=first_round_draft.get("return_version", "ty24-v1"),
                lines=[
                    DraftLine(
                        form=l.get("form", "1040"),
                        line=l.get("line", ""),
                        description=l.get("description", ""),
                        amount=_safe_amount(l.get("amount", 0.0)),
                        rationale=l.get("rationale", ""),
                    )
                    for l in first_round_draft.get("lines", [])
                ],
                metadata=first_round_draft.get("metadata", {}),
            )
            eval_before = evaluate_draft_fn(initial_draft, output_xml_path)
            correct_before_pct = eval_before.correct_by_line_score * 100

        eval_after = evaluate_draft_fn(final_draft, output_xml_path)
        correct_after_pct = eval_after.correct_by_line_score * 100

        log("## Evaluation (vs TaxCalcBench expected output.xml)")
        log()
        if correct_before_pct is not None:
            log(f"**Correct (by line) before Verifier/Approver:** {correct_before_pct:.2f}%")
            log(f"**Correct (by line) after Verifier/Approver:** {correct_after_pct:.2f}%")
            log()
        log(eval_after.report)
        log()

        if episode_history:
            last_round = episode_history[-1]
            sc = last_round.get("safety_case") or {}
            decision = last_round.get("decision") or {}
            line_findings = sc.get("line_findings") or []
            decision_label = decision.get("decision")
            recompute_lines = {
                (c.get("form"), c.get("line"))
                for c in (
                    (decision.get("required_changes") or {}).get(
                        "lines_to_recompute", []
                    )
                )
            }
            xml_str = output_xml_path.read_text(encoding="utf-8")
            counts = {i: 0 for i in range(1, 7)}

            for line_desc, xpath in LINES_TO_XPATH.items():
                expected_value = _parse_xml_value(xml_str, xpath)
                generated_value = _draft_amount_for_line(final_draft, line_desc)
                filer_correct = generated_value == expected_value

                line_prefix = line_desc.split(":")[0].strip()
                line_id = line_prefix.split(" ", 1)[1].strip() if " " in line_prefix else line_prefix

                lf = next(
                    (lf for lf in line_findings if lf.get("line") == line_id), None
                )
                v_has_sac = lf is not None
                key = ("1040", line_id)
                approver_flags_line = (
                    decision_label == "needs_revision" and key in recompute_lines
                )

                if filer_correct:
                    if not v_has_sac:
                        counts[1] += 1
                    elif approver_flags_line:
                        counts[2] += 1
                    else:
                        counts[3] += 1
                else:
                    if not v_has_sac:
                        counts[4] += 1
                    elif approver_flags_line:
                        counts[5] += 1
                    else:
                        counts[6] += 1

            log("## Per-line outcome categories (Filer / Verifier / Approver)")
            log()
            log(f"1. Filer correct, V no-sac: {counts[1]} line(s)")
            log(f"2. Filer correct, V sac, A agree: {counts[2]} line(s)")
            log(f"3. Filer correct, V sac, A disagree: {counts[3]} line(s)")
            log(f"4. Filer false, V no-sac: {counts[4]} line(s)")
            log(f"5. Filer false, V sac, A agree: {counts[5]} line(s)")
            log(f"6. Filer false, V sac, A disagree: {counts[6]} line(s)")
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
    print(f"[GPT5 Multi-Agent] Writing report to {output_path}...")
    output_path.write_text("\n".join(lines_out), encoding="utf-8")
    log()
    log(f"Output saved to: {output_path.absolute()}")
    print("[GPT5 Multi-Agent] Done.")

    return final_draft, episode_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-5 multi-agent (Filer → Verifier → Approver) showcase.")
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=None,
        help="TaxCalcBench case directory (input.json + output.xml). Default: single-retirement-1099r-alaska-dividend",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        metavar="N",
        help="Max Verifier/Approver rounds (default: 3). Use 1 for a single round.",
    )
    args = parser.parse_args()
    case_dir = Path(args.case_dir).resolve() if args.case_dir else None
    run_gpt5_multiagent_showcase(case_dir=case_dir, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()

