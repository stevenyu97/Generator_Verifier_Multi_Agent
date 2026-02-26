"""Game loop: run_episode (Filer → Verifier → Approver)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from agents import approver_agent, filer_agent, verifier_agent
from client import QwenClient
from schemas import DraftReturn


def run_episode(
    client: QwenClient,
    input_json: Dict[str, Any],
    max_rounds: int = 3,
) -> Tuple[DraftReturn, List[Dict[str, Any]]]:
    """Run one episode: Filer drafts, then up to max_rounds of Verifier + Approver; revise on needs_revision."""
    history: List[Dict[str, Any]] = []
    draft = filer_agent(client, input_json, feedback=None)

    for round_idx in range(max_rounds):
        safety_case = verifier_agent(client, input_json, draft)
        decision = approver_agent(client, input_json, draft, safety_case)

        history.append(
            {
                "round": round_idx,
                "draft": {
                    "return_version": draft.return_version,
                    "lines": [asdict(l) for l in draft.lines],
                    "metadata": draft.metadata,
                },
                "safety_case": {
                    "claim_id": safety_case.claim_id,
                    "overall_verdict": safety_case.overall_verdict,
                    "overall_confidence": safety_case.overall_confidence,
                    "line_findings": [asdict(lf) for lf in safety_case.line_findings],
                },
                "decision": asdict(decision),
            }
        )

        if decision.decision == "approve":
            break

        if decision.decision == "needs_revision":
            feedback = {
                "safety_case": history[-1]["safety_case"],
                "decision": history[-1]["decision"],
            }
            draft = filer_agent(client, input_json, feedback=feedback)

    return draft, history
