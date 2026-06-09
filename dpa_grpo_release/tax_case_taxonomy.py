"""Episode-level case taxonomy (1-6) used to weight GRPO losses per training episode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agents import parse_verifier_raw_to_safety_case
from rewards import reward_verifier_safety_case
from schemas import ApprovalDecision, DraftReturn, SafetyCase


@dataclass
class CaseTaxonomyConfig:
    x_good: float = 0.55
    verifier_good: float = 0.0
    y_prime_gt_y_margin: float = 0.05
    revise_draft_amount_abs_tol: float = 1e-2


def _line_key(form: str, line: str) -> Tuple[str, str]:
    return (form.strip().lower(), str(line).strip().lower())


def drafts_near_identical(x: DraftReturn, z: DraftReturn, *, abs_tol: float = 1e-2) -> bool:
    xm = {_line_key(l.form, l.line): float(l.amount) for l in x.lines}
    zm = {_line_key(l.form, l.line): float(l.amount) for l in z.lines}
    if xm.keys() != zm.keys():
        return False
    return all(abs(xm[k] - zm[k]) <= abs_tol for k in xm)


def build_revise_feedback_dict(
    verifier_raw: Dict[str, Any],
    draft_x: DraftReturn,
    *,
    sac_only: bool = True,
) -> Dict[str, Any]:
    sc: SafetyCase = parse_verifier_raw_to_safety_case(verifier_raw, draft_x, sac_only=sac_only)
    dec = ApprovalDecision(
        decision="needs_revision", decision_confidence=0.9, comments="", required_changes={},
    )
    return {"safety_case": asdict(sc), "decision": asdict(dec)}


def dual_verifier_rewards_from_raw(
    raw: Dict[str, Any], draft: DraftReturn, xml_path: Path,
) -> Tuple[float, float]:
    sc_sac = parse_verifier_raw_to_safety_case(raw, draft, sac_only=True)
    sc_full = parse_verifier_raw_to_safety_case(raw, draft, sac_only=False)
    return (
        reward_verifier_safety_case(sc_sac, draft, xml_path),
        reward_verifier_safety_case(sc_full, draft, xml_path),
    )


def classify_episode(
    x_acc: float,
    r_sac: float,
    r_full: float,
    *,
    z_acc: Optional[float] = None,
    revise_chose_keep_x: Optional[bool] = None,
    cfg: Optional[CaseTaxonomyConfig] = None,
) -> int:
    c = cfg or CaseTaxonomyConfig()
    x_ok = x_acc >= c.x_good
    full_ok = r_full >= c.verifier_good
    sac_ok = r_sac >= c.verifier_good
    yp_gt_y = r_full > r_sac + c.y_prime_gt_y_margin

    if z_acc is None:
        if (not x_ok) and (not full_ok):
            return 4
        if x_ok and full_ok and yp_gt_y:
            return 1
        return 0

    z_ok = z_acc >= c.x_good
    keep = revise_chose_keep_x is True
    no_keep = revise_chose_keep_x is False

    if x_ok and (not sac_ok) and full_ok and yp_gt_y and (not z_ok) and keep:
        return 2
    if x_ok and (not sac_ok) and full_ok and yp_gt_y and z_ok:
        return 3
    if (not x_ok) and sac_ok and (not full_ok) and z_ok and no_keep:
        return 5
    if (not x_ok) and sac_ok and (not full_ok) and (not z_ok):
        return 6
    return 0


def default_case_loss_weights(case_id: int) -> Tuple[float, float]:
    if case_id == 0:
        return (0.75, 0.75)
    if case_id == 4:
        return (0.35, 0.35)
    return (1.0, 1.0)
