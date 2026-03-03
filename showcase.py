"""Showcase runner: load case, run episode, save output."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from client import QwenClient
from environment import run_episode
from schemas import DraftLine, DraftReturn
from evaluator import LINES_TO_XPATH, _parse_xml_value, _draft_amount_for_line, evaluate as evaluate_draft_fn


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
            "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-retirement-1099r-alaska-dividend"
            #"/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/mfj-multiple-schedule-c-loss-multi-home-office"
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
    # Verifier evaluates each line; summarize per-line verdicts from the last round
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
        log("The Verifier should return one verdict per evaluation line (19 lines, same as TaxCalcBench). Counts from the final round:")
        if total_findings < 19:
            log(f"  _(Verifier returned only {total_findings} line finding(s); expected 19.)_")
        for v in ("correct", "plausible", "suspicious", "wrong"):
            n = by_verdict.get(v, 0)
            log(f"  - **{v}**: {n} line(s)")
        log()

        # Per-line trends: for each important line, show Verifier verdict + whether Approver flagged it
        last_decision = last_round.get("decision") or {}
        decision_label = last_decision.get("decision")
        # Set of (form, line) pairs that Approver requested to recompute
        recompute_lines = {
            (c.get("form"), c.get("line"))
            for c in (
                (last_decision.get("required_changes") or {}).get(
                    "lines_to_recompute", []
                )
            )
        }

        log("### Per-line trends (Verifier + Approver, final round)")
        log("For each important line, this shows the Verifier verdict and whether the Approver flagged the line for revision.")
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

        # Verifier & Approver interaction for this round
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

    output_xml = Path(case_dir) / "output.xml"
    if output_xml.exists():
        try:
            print("[Showcase] Evaluating draft vs output.xml...")

            # Correct (by line) BEFORE Verifier/Approver: initial Filer draft (first round)
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
                            amount=float(l.get("amount", 0)) if l.get("amount") is not None else 0.0,
                            rationale=l.get("rationale", ""),
                        )
                        for l in first_round_draft.get("lines", [])
                    ],
                    metadata=first_round_draft.get("metadata", {}),
                )
                eval_before = evaluate_draft_fn(initial_draft, output_xml)
                correct_before_pct = eval_before.correct_by_line_score * 100

            # Correct (by line) AFTER Verifier/Approver: final draft (possibly revised)
            eval_result = evaluate_draft_fn(final_draft, output_xml)
            correct_after_pct = eval_result.correct_by_line_score * 100

            log("## Evaluation (vs TaxCalcBench expected output.xml)")
            log()
            if correct_before_pct is not None:
                log(f"**Correct (by line) before Verifier/Approver:** {correct_before_pct:.2f}%")
                log(f"**Correct (by line) after Verifier/Approver:** {correct_after_pct:.2f}%")
                log()
            log(eval_result.report)
            log()

            # Categorize each important line into the 6 (Filer, Verifier, Approver) outcome types.
            # 1. Filer correct, V no-sac
            # 2. Filer correct, V sac, A agree
            # 3. Filer correct, V sac, A disagree
            # 4. Filer false, V no-sac
            # 5. Filer false, V sac, A agree
            # 6. Filer false, V sac, A disagree
            if episode_history:
                last_round = episode_history[-1]
                sc = last_round.get("safety_case") or {}
                decision = last_round.get("decision") or {}
                line_findings = sc.get("line_findings") or []
                decision_label = decision.get("decision")
                # Lines explicitly requested for recomputation by Approver
                recompute_lines = {
                    (c.get("form"), c.get("line"))
                    for c in (
                        (decision.get("required_changes") or {}).get(
                            "lines_to_recompute", []
                        )
                    )
                }

                # Pre-read XML so we can compare Filer vs ground truth per line
                xml_str = output_xml.read_text(encoding="utf-8")

                counts = {i: 0 for i in range(1, 7)}

                for line_desc, xpath in LINES_TO_XPATH.items():
                    # Ground-truth correctness (Filer correct or false)
                    expected_value = _parse_xml_value(xml_str, xpath)
                    generated_value = _draft_amount_for_line(final_draft, line_desc)
                    filer_correct = generated_value == expected_value

                    # Map "Line 9: ..." → "9"
                    line_prefix = line_desc.split(":")[0].strip()  # e.g. "Line 9"
                    if " " in line_prefix:
                        line_id = line_prefix.split(" ", 1)[1].strip()
                    else:
                        line_id = line_prefix

                    # Verifier safety assurance case (sac) for this line?
                    lf = next(
                        (lf for lf in line_findings if lf.get("line") == line_id), None
                    )
                    v_has_sac = lf is not None

                    # Approver per-line stance: did Approver request this line to be recomputed?
                    key = ("1040", line_id)
                    approver_flags_line = (
                        decision_label == "needs_revision" and key in recompute_lines
                    )

                    # For lines with a safety case, "A agree" means Approver also treats the line as needing revision.
                    if filer_correct:
                        if not v_has_sac:
                            counts[1] += 1  # Filer correct, V no-sac
                        else:
                            if approver_flags_line:
                                counts[2] += 1  # Filer correct, V sac, A agree
                            else:
                                counts[3] += 1  # Filer correct, V sac, A disagree
                    else:
                        if not v_has_sac:
                            counts[4] += 1  # Filer false, V no-sac
                        else:
                            if approver_flags_line:
                                counts[5] += 1  # Filer false, V sac, A agree
                            else:
                                counts[6] += 1  # Filer false, V sac, A disagree

                log("## Per-line outcome categories (Filer / Verifier / Approver)")
                log()
                log(
                    f"1. Filer correct, V no-sac: {counts[1]} line(s)  (V no-sac only occurs when V judged the line correct)."
                )
                log(
                    f"2. Filer correct, V sac, A agree: {counts[2]} line(s)"
                )
                log(
                    f"3. Filer correct, V sac, A disagree: {counts[3]} line(s)"
                )
                log(
                    f"4. Filer false, V no-sac: {counts[4]} line(s)"
                )
                log(
                    f"5. Filer false, V sac, A agree: {counts[5]} line(s)"
                )
                log(
                    f"6. Filer false, V sac, A disagree: {counts[6]} line(s)"
                )
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
