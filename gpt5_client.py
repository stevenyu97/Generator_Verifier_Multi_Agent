import json
import os
from typing import Any, Dict

from litellm import responses

from client import extract_json_from_response
from gpt5_filer import GPT5_MODEL, REASONING_EFFORT


class Gpt5Client:
    """
    Drop-in replacement for QwenClient that uses GPT-5 via LiteLLM Responses API.

    Exposes the same interface (chat / generate_json) so it can be passed into
    filer_agent, verifier_agent, approver_agent, and run_episode.
    """

    def __init__(
        self,
        model: str = GPT5_MODEL,
        reasoning_effort: str = REASONING_EFFORT,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort

    def _call_responses(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int = 32768,
    ) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        full_prompt = f"{system_prompt}\n\n---\n\nUser input:\n{user_content}"

        response = responses(
            model=self.model,
            input=full_prompt,
            reasoning={"effort": self.reasoning_effort},
        )

        # Extract assistant message text from response.output (Responses API format)
        text = ""
        for entry in response.output:
            if getattr(entry, "type", None) == "message" and getattr(
                entry, "content", None
            ):
                # content is a list of parts; take the first text part
                part = entry.content[0]
                text = getattr(part, "text", "") or ""
                break
        return text

    def chat(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Run a single-turn chat and return the raw text response."""
        return self._call_responses(system_prompt, user_content, max_new_tokens)

    def generate_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        max_new_tokens: int = 32768,
    ) -> Dict[str, Any]:
        """Call GPT-5 and parse a JSON object from the response."""
        user_text = json.dumps(user_payload, indent=2)
        assistant = self._call_responses(system_prompt, user_text, max_new_tokens)
        data = extract_json_from_response(assistant)
        if data is None:
            raise ValueError(f"Model did not return JSON: {assistant[:500]}...")
        return data

