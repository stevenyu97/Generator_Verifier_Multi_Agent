"""
Per-line six-case taxonomy and episode scores for GRPO.

**Convention**
- **x / r_x**: filer line correctness vs gold.
- **y / r_y**: verifier provides SAC for the line (non-silent).
- **y′ / r_y′**: verifier is silent for the line.
- **z / z′**: from one revise draft **z** (same episode); **per line**:
  - **r_z[l]**: revision **helps** this line (touched line, x wrong, z right vs gold).
  - **r_z′[l]**: **keeping x** (or rejecting a harmful z) is the right signal on this line.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents import parse_verifier_raw_to_safety_case
from line_eval import (
    draft_amount_for_line_id,
    line_strict_correctness_by_line_id,
    reward_filer_draft,
)
from rewards import verdict_by_line_from_case
from schemas import DraftReturn, SafetyCase


@dataclass
class LineCaseConfig:
    """Per-line z / z′ and optional global z_acc check."""

    z_acc_improve_eps: float = 1e-4
    line_amount_tol: float = 1e-6


def per_line_rz_rzp(
    draft_x: DraftReturn,
    z: Optional[DraftReturn],
    xml_path: Path,
    *,
    line_cfg: LineCaseConfig,
    draft_identical_abs_tol: float,
) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]:
    """
    Per-line **z** (revision helps line) and **z′** (keeping / preferring x on the line is right).

    **touched[l]**: ``z`` exists, drafts differ on that line's amount vs ``x`` by more than
    ``line_cfg.line_amount_tol`` (and global near-identical drafts ⇒ no line touched).
    """
    from tax_case_taxonomy import drafts_near_identical

    correctness_x = line_strict_correctness_by_line_id(draft_x, xml_path)
    if z is None:
        rz = {lid: False for lid in correctness_x}
        rzp = {lid: bool(correctness_x[lid]) for lid in correctness_x}
        touched = {lid: False for lid in correctness_x}
        return rz, rzp, touched

    if drafts_near_identical(draft_x, z, abs_tol=draft_identical_abs_tol):
        rz = {lid: False for lid in correctness_x}
        rzp = {lid: bool(correctness_x[lid]) for lid in correctness_x}
        touched = {lid: False for lid in correctness_x}
        return rz, rzp, touched

    correctness_z = line_strict_correctness_by_line_id(z, xml_path)
    tol = line_cfg.line_amount_tol
    rz: Dict[str, bool] = {}
    rzp: Dict[str, bool] = {}
    touched: Dict[str, bool] = {}

    for line_id, x_ok in correctness_x.items():
        ax = draft_amount_for_line_id(draft_x, line_id)
        az = draft_amount_for_line_id(z, line_id)
        t = abs(ax - az) > tol
        z_ok = correctness_z.get(line_id, False)
        touched[line_id] = t
        if not t:
            rz[line_id] = False
            rzp[line_id] = x_ok
            continue
        # Revision changed this line's amount vs x.
        rz[line_id] = (not x_ok) and z_ok
        if x_ok and (not z_ok):
            rzp[line_id] = True
        elif (not x_ok) and z_ok:
            rzp[line_id] = False
        elif (not x_ok) and (not z_ok):
            rzp[line_id] = False
        else:
            # x_ok and z_ok: both match gold; amount diff shouldn't happen if gold unique — prefer keep.
            rzp[line_id] = True

    return rz, rzp, touched


def classify_line_case_id(
    *,
    rx: bool,
    ry: bool,
    ryp: bool,
    rz: bool,
    rzp: bool,
    touched: bool,
) -> int:
    """Assign one of 1–6 using per-line x, y (SAC present), y′ (silent), z, z′.

    Case meanings aligned with the user convention:
    1. x correct, verifier silent
    2. x correct, verifier SAC, non-case-3 path (e.g., revision pressure / touched line)
    3. x correct, verifier SAC, keep-x right and no revision on x (z' chosen)
    4. x incorrect, verifier silent
    5. x incorrect, verifier SAC, revision helps
    6. x incorrect, verifier SAC, revision does not help
    """
    # Case 1: x correct, verifier silent.
    if rx and ryp:
        return 1
    # Case 3: x correct, verifier SAC present, keep-x is right, and no revision was applied on x.
    if rx and ry and rzp and (not touched):
        return 3
    # Case 2: remaining x-correct + SAC-present paths.
    if rx and ry:
        return 2
    # Case 4: x incorrect, verifier silent.
    if (not rx) and ryp:
        return 4
    # Case 5/6: x incorrect, verifier SAC present.
    if (not rx) and ry and rz:
        return 5
    if (not rx) and ry and (not rz):
        return 6

    # Total-classification fallbacks: ensure every line maps into one of 1..6.
    if rx and (not ry):
        return 1 if ryp else 3
    if rx and ry:
        return 3 if (rzp and not touched) else 2
    if (not rx) and (not ry):
        return 4
    if (not rx) and ry:
        return 5 if rz else 6
    return 6


def per_line_case_counts_from_verifier_raw(
    draft_x: DraftReturn,
    verifier_raw: Dict[str, Any],
    xml_path: Path,
    z_draft: Optional[DraftReturn],
    z_acc: float,
    *,
    revise_abs_tol: float,
    line_cfg: LineCaseConfig,
) -> Tuple[Dict[int, int], List[Tuple[str, int]], float]:
    """
    Returns:
    - counts: case_id -> count
    - line_ids: list of (line_id, case_id) for evaluated lines
    - x_acc (global filer scalar) for logging
    """
    sc_sac: SafetyCase = parse_verifier_raw_to_safety_case(
        verifier_raw, draft_x, sac_only=True
    )
    sac_map = verdict_by_line_from_case(sc_sac)
    correctness = line_strict_correctness_by_line_id(draft_x, xml_path)
    x_acc = reward_filer_draft(draft_x, xml_path)

    rz_map, rzp_map, touched_map = per_line_rz_rzp(
        draft_x,
        z_draft,
        xml_path,
        line_cfg=line_cfg,
        draft_identical_abs_tol=revise_abs_tol,
    )

    counts: Dict[int, int] = {i: 0 for i in range(1, 7)}
    line_rows: List[Tuple[str, int]] = []
    for line_id, _gold_ok in correctness.items():
        rx = bool(correctness[line_id])
        verdict = sac_map.get(line_id, "missing")
        ry = verdict not in ("missing", "", None)
        ryp = not ry
        cid = classify_line_case_id(
            rx=rx,
            ry=ry,
            ryp=ryp,
            rz=rz_map.get(line_id, False),
            rzp=rzp_map.get(line_id, False),
            touched=touched_map.get(line_id, False),
        )
        if 1 <= cid <= 6:
            counts[cid] = counts.get(cid, 0) + 1
        line_rows.append((line_id, cid))

    return counts, line_rows, float(x_acc)


def case_preference_score(counts: Dict[int, int]) -> float:
    """In [-1, 1]: favor cases 1 and 5, penalize 2–4 and 6. Case 0 ignored in numerator."""
    prefer = counts.get(1, 0) + counts.get(5, 0)
    penal = (
        counts.get(2, 0)
        + counts.get(3, 0)
        + counts.get(4, 0)
        + counts.get(6, 0)
    )
    n = prefer + penal
    if n <= 0:
        return 0.0
    return float(prefer - penal) / float(n)


def line_case_loss_weights(
    counts: Dict[int, int],
    *,
    base_wf: float,
    base_wv: float,
    prefer_boost: float,
) -> Tuple[float, float]:
    """Scale base (filer, verifier) weights by episode case-mix preference."""
    s = case_preference_score(counts)
    m = 1.0 + prefer_boost * s
    m = max(0.25, min(1.75, m))
    return base_wf * m, base_wv * m
