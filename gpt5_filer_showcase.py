import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from prompts import FILER_SYSTEM_PROMPT
from schemas import DraftLine, DraftReturn
from evaluator import evaluate as evaluate_draft


client = OpenAI()

# Default TaxCalcBench case directory (edit this if you want a different default)
DEFAULT_CASE_DIR = Path(
    "/home/ubuntu/LLM/Dataset/taxcalcbench_dataset/test_data/single-retirement-1099r-alaska-dividend"
)


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("Model did not return JSON")
    return json.loads(text[first:last + 1])


def run_gpt5_filer(input_json: dict, max_new_tokens: int = 32768) -> DraftReturn:
    """
    Call GPT-5 with the same Filer prompt and payload shape as the local filer_agent.
    Returns a DraftReturn compatible with the evaluator.
    """
    client.api_key = os.environ.get("OPENAI_API_KEY", client.api_key)
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    payload = {"input": input_json}
    user_text = json.dumps(payload, indent=2)

    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": FILER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    raw = extract_json_from_response(content)

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
    print()
    print(f"Strictly correct return: {eval_result.strictly_correct_return}")
    print(f"Lenient correct return: {eval_result.lenient_correct_return}")
    print(f"Correct (by line): {eval_result.correct_by_line_score * 100:.2f}%")
    print(f"Correct (by line, lenient): {eval_result.lenient_correct_by_line_score * 100:.2f}%")


if __name__ == "__main__":
    main()

