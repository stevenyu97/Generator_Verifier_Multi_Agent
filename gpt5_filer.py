import json
import os
import sys

from litellm import responses

from prompts import FILER_SYSTEM_PROMPT

# Match TaxCalcBench: same model and reasoning API for comparable accuracy (~80% by line)
GPT5_MODEL = "openai/gpt-5-2025-08-07"
REASONING_EFFORT = "high"


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("Model did not return JSON")
    return json.loads(text[first:last + 1])


def run_filer(input_json: dict, max_new_tokens: int = 32768) -> dict:
    """
    Call GPT-5 via LiteLLM Responses API with reasoning (same as TaxCalcBench)
    so accuracy is comparable (~80% correct by line). Returns raw JSON with
    `return_version`, `lines`, and `metadata`.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    payload = {"input": input_json}
    user_text = json.dumps(payload, indent=2)
    # Single prompt string as in TaxCalcBench (responses API uses "input", not messages)
    full_prompt = f"{FILER_SYSTEM_PROMPT}\n\n---\n\nUser input:\n{user_text}"

    response = responses(
        model=GPT5_MODEL,
        input=full_prompt,
        reasoning={"effort": REASONING_EFFORT},
    )
    # Extract assistant message text from response output (same as tax_return_generator)
    content = ""
    for entry in response.output:
        if getattr(entry, "type", None) == "message" and getattr(entry, "content", None):
            content = entry.content[0].text
            break
    return extract_json_from_response(content or "")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python gpt5_filer.py path/to/input.json", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    payload = data.get("input", data)
    result = run_filer(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

