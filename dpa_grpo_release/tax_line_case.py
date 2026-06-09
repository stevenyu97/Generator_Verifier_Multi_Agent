"""Per-line 6-case taxonomy used to bias GRPO rewards toward cases 1 and 5.

Signals (per evaluated line l):
  rx  := filer line correct vs gold
  ry  := verifier raised SAC for l (non-silent)
  ry' := verifier silent for l
  rz  := revision z helps l (touched, x wrong, z right)
  rz' := keeping x (or rejecting harmful z) is the right call on l
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents import parse_verifier_raw_to_safety_case
from evaluator import LINES_TO_XPATH, _draft_amount_for_line, line_strict_correctness_by_line_id
from rewards import reward_filer_draft, verdict_by_line_from_case
from schemas import DraftReturn, SafetyCase


@dataclass
class LineCaseConfig:
    z_acc_improve_eps: float = 1e-4
    line_amount_tol: float = 1e-6


def _line_desc_for_id(line_id: str) -> Optional[str]:
    for ld in LINES_TO_XPATH:
        prefix = ld.split(":")[0].strip()
        lid = prefix.replace("Line ", "").strip() if prefix.startswith("Line ") else prefix
        if lid == line_id:
            return ld
    return None


def per_line_rz_rzp(
    draft_x: DraftReturn,
    z: Optional[DraftReturn],
    xml_path: Path,
    *,
    line_cfg: LineCaseConfig,
    draft_identical_abs_tol: float,
) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]:
    from tax_case_taxonomy import drafts_near_identical

    cx = line_strict_correctness_by_line_id(draft_x, xml_path)
    if z is None or drafts_near_identical(draft_x, z, abs_tol=draft_identical_abs_tol):
        return (
            {lid: False for lid in cx},
            {lid: bool(cx[lid]) for lid in cx},
            {lid: False for lid in cx},
        )

    cz = line_strict_correctness_by_line_id(z, xml_path)
    tol = line_cfg.line_amount_tol
    rz: Dict[str, bool] = {}
    rzp: Dict[str, bool] = {}
    touched: Dict[str, bool] = {}
    for lid, x_ok in cx.items():
        ld = _line_desc_for_id(lid)
        if ld is None:
            rz[lid], rzp[lid], touched[lid] = False, x_ok, False
            continue
        ax, az = _draft_amount_for_line(draft_x, ld), _draft_amount_for_line(z, ld)
        t = abs(ax - az) > tol
        z_ok = cz.get(lid, False)
        touched[lid] = t
        if not t:
            rz[lid], rzp[lid] = False, x_ok
            continue
        rz[lid] = (not x_ok) and z_ok
        if x_ok and (not z_ok):
            rzp[lid] = True
        elif (not x_ok) and z_ok:
            rzp[lid] = False
        elif (not x_ok) and (not z_ok):
            rzp[lid] = False
        else:
            rzp[lid] = True
    return rz, rzp, touched


def classify_line_case_id(
    *, rx: bool, ry: bool, ryp: bool, rz: bool, rzp: bool, touched: bool,
) -> int:
    if rx and ryp:
        return 1
    if rx and ry and rzp and (not touched):
        return 3
    if rx and ry:
        return 2
    if (not rx) and ryp:
        return 4
    if (not rx) and ry and rz:
        return 5
    if (not rx) and ry and (not rz):
        return 6
    if rx and (not ry):
        return 1 if ryp else 3
    if (not rx) and (not ry):
        return 4
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
    sc_sac: SafetyCase = parse_verifier_raw_to_safety_case(verifier_raw, draft_x, sac_only=True)
    sac_map = verdict_by_line_from_case(sc_sac)
    correctness = line_strict_correctness_by_line_id(draft_x, xml_path)
    x_acc = reward_filer_draft(draft_x, xml_path)
    rz_map, rzp_map, touched_map = per_line_rz_rzp(
        draft_x, z_draft, xml_path,
        line_cfg=line_cfg, draft_identical_abs_tol=revise_abs_tol,
    )
    counts: Dict[int, int] = {i: 0 for i in range(1, 7)}
    rows: List[Tuple[str, int]] = []
    for lid in correctness:
        rx = bool(correctness[lid])
        verdict = sac_map.get(lid, "missing")
        ry = verdict not in ("missing", "", None)
        cid = classify_line_case_id(
            rx=rx, ry=ry, ryp=not ry,
            rz=rz_map.get(lid, False),
            rzp=rzp_map.get(lid, False),
            touched=touched_map.get(lid, False),
        )
        if 1 <= cid <= 6:
            counts[cid] = counts.get(cid, 0) + 1
        rows.append((lid, cid))
    return counts, rows, float(x_acc)


def case_preference_score(counts: Dict[int, int]) -> float:
    """In [-1, 1]: cases 1 & 5 are good outcomes; 2, 3, 4, 6 are not."""
    prefer = counts.get(1, 0) + counts.get(5, 0)
    penal = counts.get(2, 0) + counts.get(3, 0) + counts.get(4, 0) + counts.get(6, 0)
    n = prefer + penal
    return float(prefer - penal) / float(n) if n > 0 else 0.0


def line_case_loss_weights(
    counts: Dict[int, int], *, base_wf: float, base_wv: float, prefer_boost: float,
) -> Tuple[float, float]:
    s = case_preference_score(counts)
    m = max(0.25, min(1.75, 1.0 + prefer_boost * s))
    return base_wf * m, base_wv * m
