import json
import os
import sys

from openai import OpenAI

from prompts import FILER_SYSTEM_PROMPT


client = OpenAI()


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("Model did not return JSON")
    return json.loads(text[first:last + 1])


def run_filer(input_json: dict, max_new_tokens: int = 32768) -> dict:
    """
    Call a GPT-5 model with the same Filer prompt used in the local pipeline.
    Returns the raw JSON object with `return_version`, `lines`, and `metadata`.
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
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    return extract_json_from_response(content)


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

