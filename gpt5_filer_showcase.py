import json
import sys
from pathlib import Path

from gpt5_filer import run_filer
from schemas import DraftLine, DraftReturn
from evaluator import evaluate as evaluate_draft

# Default TaxCalcBench case directory (edit this if you want a different default)
DEFAULT_CASE_DIR = Path(
    "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-retirement-1099r-alaska-dividend"
    #"/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/mfj-multiple-schedule-c-loss-multi-home-office"
)


def run_gpt5_filer(input_json: dict, max_new_tokens: int = 32768) -> DraftReturn:
    """
    Call GPT-5 via gpt5_filer (LiteLLM Responses API + reasoning, same as TaxCalcBench).
    Returns a DraftReturn compatible with the evaluator.
    """
    raw = run_filer(input_json)

    lines = []
    for l in raw.get("lines", []):
        # Be robust to models that emit amount as "", "0", or other non-float types.
        raw_amount = l.get("amount", 0.0)
        try:
            amount_val = float(raw_amount)
        except (TypeError, ValueError):
            amount_val = 0.0
        lines.append(
            DraftLine(
                form=l.get("form", "1040"),
                line=l.get("line", ""),
                description=l.get("description", ""),
                amount=amount_val,
                rationale=l.get("rationale", ""),
            )
        )

    return DraftReturn(
        return_version=raw.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=raw.get("metadata", {}),
    )


def load_taxcalcbench_case(case_dir: Path) -> dict:
    input_path = case_dir / "input.json"
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)


def main() -> None:
    if len(sys.argv) == 2:
        case_dir = Path(sys.argv[1]).resolve()
    else:
        # Fall back to the default case directory when no argument is provided
        case_dir = DEFAULT_CASE_DIR
    input_path = case_dir / "input.json"
    output_xml_path = case_dir / "output.xml"

    if not input_path.exists():
        print(f"input.json not found in {case_dir}", file=sys.stderr)
        sys.exit(1)
    if not output_xml_path.exists():
        print(f"output.xml not found in {case_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[GPT5 Filer] Loading input case from {case_dir}...")
    input_json = load_taxcalcbench_case(case_dir)

    print("[GPT5 Filer] Calling GPT-5 to generate draft...")
    draft = run_gpt5_filer(input_json)
    print(f"[GPT5 Filer] Draft generated with {len(draft.lines)} lines.")

    print("[GPT5 Filer] Evaluating draft vs output.xml (same metrics as showcase)...")
    eval_result = evaluate_draft(draft, output_xml_path)

    print()
    print("## Input case")
    print(f"Case directory: `{case_dir}`")
    print()
    print("## Evaluation (vs TaxCalcBench expected output.xml)")
    print()
    print(eval_result.report)


if __name__ == "__main__":
    main()

