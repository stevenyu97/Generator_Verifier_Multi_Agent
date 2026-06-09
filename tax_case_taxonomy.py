"""
Six-case training guidance: oracle signals → case id 1–6 → loss weights.

**Signals**
- ``x_acc``: ``reward_filer_draft`` in [0, 1] for the conditioning draft ``x``.
- **y** (SAC) vs **y'** (full): same verifier JSON, ``parse_verifier_raw_to_safety_case``
  with ``sac_only=True/False``, then ``reward_verifier_safety_case`` → ``r_sac``, ``r_full``.

**Revision** ``z`` / keep ``x``: training can run one extra filer sample with verifier JSON as
``feedback`` (see ``build_revise_feedback_dict``). Pass ``z_acc = reward_filer_draft(z)`` and
``revise_chose_keep_x = drafts_near_identical(x, z)``. Without that, cases 2–3, 5–6 usually
map to **0** (unknown).
"""
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


def drafts_near_identical(
    x: DraftReturn,
    z: DraftReturn,
    *,
    abs_tol: float = 1e-2,
) -> bool:
    """True if revised draft ``z`` matches ``x`` on every (form, line) key and amount (oracle-free)."""
    xm = {_line_key(l.form, l.line): float(l.amount) for l in x.lines}
    zm = {_line_key(l.form, l.line): float(l.amount) for l in z.lines}
    if xm.keys() != zm.keys():
        return False
    for k, va in xm.items():
        if abs(va - zm[k]) > abs_tol:
            return False
    return True


def build_revise_feedback_dict(
    verifier_raw: Dict[str, Any],
    draft_x: DraftReturn,
    *,
    sac_only: bool = True,
) -> Dict[str, Any]:
    """Payload for ``filer_agent``-style revise: verifier findings + fixed ``needs_revision`` decision."""
    sc: SafetyCase = parse_verifier_raw_to_safety_case(
        verifier_raw, draft_x, sac_only=sac_only
    )
    dec = ApprovalDecision(
        decision="needs_revision",
        decision_confidence=0.9,
        comments="",
        required_changes={},
    )
    return {"safety_case": asdict(sc), "decision": asdict(dec)}


def dual_verifier_rewards_from_raw(
    raw: Dict[str, Any],
    draft: DraftReturn,
    xml_path: Path,
) -> Tuple[float, float]:
    sc_sac = parse_verifier_raw_to_safety_case(raw, draft, sac_only=True)
    sc_full = parse_verifier_raw_to_safety_case(raw, draft, sac_only=False)
    r_sac = reward_verifier_safety_case(sc_sac, draft, xml_path)
    r_full = reward_verifier_safety_case(sc_full, draft, xml_path)
    return r_sac, r_full


def classify_episode(
    x_acc: float,
    r_sac: float,
    r_full: float,
    *,
    z_acc: Optional[float] = None,
    revise_chose_keep_x: Optional[bool] = None,
    cfg: Optional[CaseTaxonomyConfig] = None,
) -> int:
    """
    Return case **1–6** or **0** (unknown / insufficient signals for z-cases).

    Without ``z_acc``, **1** and **4** are resolved when indicators match; **2,3,5,6** need
    revision data or map to **0**.
    """
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
    """``(filer_weight, verifier_weight)`` multipliers for GRPO terms."""
    if case_id == 0:
        return (0.75, 0.75)
    if case_id == 1:
        return (1.0, 1.0)
    if case_id in (2, 3, 5, 6):
        return (1.0, 1.0)
    if case_id == 4:
        return (0.35, 0.35)
    return (1.0, 1.0)
