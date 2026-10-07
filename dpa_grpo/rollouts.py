"""DPA-GRPO: rollouts module."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.agents import _trim_draft_for_verifier
from core.client import extract_json_from_response
from core.rewards import (
    line_verdict_binary_aligned,
    reward_filer_draft,
    reward_verifier_safety_case,
)
from core.schemas import DraftLine, DraftReturn
from benchmarks.line_eval import evaluated_line_ids, line_strict_correctness_by_line_id
from benchmarks.tax.tax_case_taxonomy import (
    CaseTaxonomyConfig,
    build_revise_feedback_dict,
    drafts_near_identical,
)
from benchmarks.tax.tax_line_case import classify_line_case_id
from dpa_grpo.config import (
    _ds_cfg,
    _filer_line_prompt,
    _line_filer_user_payload,
    _line_revise_user_payload,
    _line_verifier_user_payload,
    _verifier_system_prompt_for,
)
from dpa_grpo.data import _draft_to_dict
from dpa_grpo.parsing import (
    _extract_line_verdict_from_raw,
    build_chat_text,
    try_parse_draft,
    try_parse_line_draft,
    try_parse_verifier_safety_case,
)
from dpa_grpo.sampling import generate_group_sequences



def filer_grpo_rollouts(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    gold_path: Path,
    args: argparse.Namespace,
    device: str,
) -> Tuple[
    List[torch.Tensor],
    int,
    torch.Tensor,
    torch.Tensor,
    List[Optional[DraftReturn]],
    List[str],
]:
    """Sample K filer completions; rewards vs xml; return sequences + advantages + drafts."""
    filer_user = json.dumps({"input": input_json, "feedback": None}, indent=2)
    filer_text = build_chat_text(tokenizer, _ds_cfg().filer_system_prompt, filer_user)
    enc_f = tokenizer(
        filer_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    pid_f = enc_f["input_ids"].to(device)
    mask_f = enc_f["attention_mask"].to(device)
    prompt_len_f = int(pid_f.shape[1])

    seqs_f, _ = generate_group_sequences(
        model,
        tokenizer,
        pid_f,
        mask_f,
        args.group_size,
        args.filer_max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )

    texts_f = [
        tokenizer.decode(s[0, prompt_len_f:], skip_special_tokens=True) for s in seqs_f
    ]
    rewards_f: List[float] = []
    drafts: List[Optional[DraftReturn]] = []
    for t in texts_f:
        d = try_parse_draft(t)
        drafts.append(d)
        if d is None:
            rewards_f.append(args.invalid_json_reward)
        else:
            rewards_f.append(reward_filer_draft(d, gold_path))

    r_f = torch.tensor(rewards_f, device=device, dtype=torch.float32)
    adv_f = (r_f - r_f.mean()) / (r_f.std(unbiased=False) + 1e-8)
    return seqs_f, prompt_len_f, adv_f, r_f, drafts, texts_f

def verifier_grpo_rollouts(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    draft_x: DraftReturn,
    gold_path: Path,
    args: argparse.Namespace,
    device: str,
) -> Tuple[List[torch.Tensor], int, torch.Tensor, torch.Tensor, List[str]]:
    """Sample K verifier completions; return sequences + advantages + rewards + raw texts."""
    v_payload = {
        "input": _ds_cfg().trim_input_for_verifier(input_json),
        "draft_return": _trim_draft_for_verifier(draft_x),
    }
    ver_user = json.dumps(v_payload, indent=2)
    ver_text = build_chat_text(tokenizer, _ds_cfg().verifier_system_prompt, ver_user)
    enc_v = tokenizer(
        ver_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    pid_v = enc_v["input_ids"].to(device)
    mask_v = enc_v["attention_mask"].to(device)
    prompt_len_v = int(pid_v.shape[1])

    seqs_v, _ = generate_group_sequences(
        model,
        tokenizer,
        pid_v,
        mask_v,
        args.group_size,
        args.verifier_max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )

    texts_v = [
        tokenizer.decode(s[0, prompt_len_v:], skip_special_tokens=True) for s in seqs_v
    ]
    rewards_v: List[float] = []
    for t in texts_v:
        sc = try_parse_verifier_safety_case(t, draft_x)
        if sc is None:
            rewards_v.append(args.invalid_json_reward)
        else:
            rewards_v.append(reward_verifier_safety_case(sc, draft_x, gold_path))

    r_v = torch.tensor(rewards_v, device=device, dtype=torch.float32)
    adv_v = (r_v - r_v.mean()) / (r_v.std(unbiased=False) + 1e-8)
    return seqs_v, prompt_len_v, adv_v, r_v, texts_v

def linewise_filer_verifier_rollouts(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    gold_path: Path,
    args: argparse.Namespace,
    device: str,
) -> Tuple[
    List[torch.Tensor],
    List[int],
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[int],
    torch.Tensor,
    torch.Tensor,
    DraftReturn,
    Dict[int, int],
    List[Dict[str, Any]],
]:
    """Line-by-line filer then verifier rollouts; aggregate all line groups for GRPO."""
    line_ids = list(evaluated_line_ids(gold_path))
    seqs_f_all: List[torch.Tensor] = []
    plen_f_all: List[int] = []
    rewards_f_all: List[float] = []
    selected_lines: List[DraftLine] = []
    context_so_far: List[Dict[str, Any]] = []
    line_details: List[Dict[str, Any]] = []

    for lid in line_ids:
        user_text = _line_filer_user_payload(input_json, lid, context_so_far)
        ptxt = build_chat_text(tokenizer, _filer_line_prompt(), user_text)
        enc = tokenizer(ptxt, return_tensors="pt", add_special_tokens=False)
        pid = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        plen = int(pid.shape[1])
        seqs, _ = generate_group_sequences(
            model,
            tokenizer,
            pid,
            mask,
            args.group_size,
            args.filer_max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        cands: List[Optional[DraftLine]] = []
        texts_line: List[str] = []
        rs: List[float] = []
        for s in seqs:
            text = tokenizer.decode(s[0, plen:], skip_special_tokens=True)
            texts_line.append(text)
            dl = try_parse_line_draft(text, lid)
            cands.append(dl)
            if dl is None:
                rs.append(args.invalid_json_reward)
            else:
                dtmp = DraftReturn(return_version=_ds_cfg().return_version, lines=[dl], metadata={})
                ok = line_strict_correctness_by_line_id(dtmp, gold_path).get(lid, False)
                rs.append(1.0 if ok else 0.0)
        bi = int(max(range(len(rs)), key=lambda i: rs[i]))
        if cands[bi] is None:
            selected_lines.append(
                DraftLine(
                    form=_ds_cfg().form_name,
                    line=lid,
                    description=f"Line {lid}",
                    amount=0.0,
                    rationale="fallback_invalid_json",
                )
            )
        else:
            selected_lines.append(cands[bi])  # type: ignore[arg-type]
        sline = selected_lines[-1]
        line_details.append(
            {
                "line": lid,
                "filer_candidate_texts": texts_line,
                "filer_candidate_rewards": rs,
                "filer_selected_idx": bi,
                "filer_selected_json": _draft_to_dict(
                    DraftReturn(return_version=_ds_cfg().return_version, lines=[sline], metadata={})
                ),
            }
        )
        context_so_far.append(
            {
                "form": sline.form,
                "line": sline.line,
                "description": sline.description,
                "amount": sline.amount,
                "rationale": sline.rationale,
            }
        )
        seqs_f_all.extend(seqs)
        plen_f_all.extend([plen] * len(seqs))
        rewards_f_all.extend(rs)

    draft_x = DraftReturn(return_version=_ds_cfg().return_version, lines=selected_lines, metadata={})
    gold = line_strict_correctness_by_line_id(draft_x, gold_path)

    seqs_v_all: List[torch.Tensor] = []
    plen_v_all: List[int] = []
    rewards_v_all: List[float] = []
    counts: Dict[int, int] = {i: 0 for i in range(1, 7)}
    for dl in selected_lines:
        lid = str(dl.line).strip()
        gold_ok = bool(gold.get(lid, False))
        prior_ctx = [x for x in context_so_far if str(x.get("line", "")) != lid]
        user_text = _line_verifier_user_payload(input_json, dl, lid, prior_ctx)
        v_sys_prompt = _verifier_system_prompt_for(args)
        ptxt = build_chat_text(tokenizer, v_sys_prompt, user_text)
        enc = tokenizer(ptxt, return_tensors="pt", add_special_tokens=False)
        pid = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        plen = int(pid.shape[1])
        seqs, _ = generate_group_sequences(
            model,
            tokenizer,
            pid,
            mask,
            args.group_size,
            args.verifier_max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        ver_texts_line: List[str] = []
        for s in seqs:
            text = tokenizer.decode(s[0, plen:], skip_special_tokens=True)
            ver_texts_line.append(text)
            raw = extract_json_from_response(text.strip())
            if not raw:
                rewards_v_all.append(args.invalid_json_reward)
            else:
                verdict = _extract_line_verdict_from_raw(raw, lid)
                good = line_verdict_binary_aligned(verdict, gold_ok)
                rewards_v_all.append(1.0 if good else 0.0)
        seqs_v_all.extend(seqs)
        plen_v_all.extend([plen] * len(seqs))
        # Taxonomy from best verifier sample for this line:
        k = len(seqs)
        rs_line = rewards_v_all[-k:]
        bi = int(max(range(k), key=lambda i: rs_line[i]))
        s_best = seqs[bi]
        text_best = tokenizer.decode(s_best[0, plen:], skip_special_tokens=True)
        raw_best = extract_json_from_response(text_best.strip()) or {}
        verdict_full = _extract_line_verdict_from_raw(raw_best, lid)
        # Updated semantics:
        # ry      := verifier provides SAC for this line (non-silent)
        # ry_prime:= verifier silent for this line
        ry = verdict_full not in ("missing", "", None)
        ryp = not ry
        rx = gold_ok
        rz = False
        rzp = rx
        touched = False
        # True linewise revision: generate z_l conditioned on verifier feedback for this line.
        if raw_best:
            r_user = _line_revise_user_payload(input_json, dl, lid, raw_best, prior_ctx)
            rtxt = build_chat_text(tokenizer, _filer_line_prompt(), r_user)
            renc = tokenizer(rtxt, return_tensors="pt", add_special_tokens=False)
            rpid = renc["input_ids"].to(device)
            rmask = renc["attention_mask"].to(device)
            rplen = int(rpid.shape[1])
            rseqs, _ = generate_group_sequences(
                model,
                tokenizer,
                rpid,
                rmask,
                1,
                args.filer_max_new_tokens,
                args.temperature,
                args.top_p,
                device,
            )
            rtext = tokenizer.decode(rseqs[0][0, rplen:], skip_special_tokens=True)
            z_line = try_parse_line_draft(rtext, lid)
            if z_line is not None:
                # touched := revised amount differs from x on this line.
                touched = abs(float(z_line.amount) - float(dl.amount)) > float(
                    args.line_case_amount_tol
                )
                if touched:
                    ztmp = DraftReturn(return_version=_ds_cfg().return_version, lines=[z_line], metadata={})
                    z_ok = line_strict_correctness_by_line_id(ztmp, gold_path).get(lid, False)
                    rz = (not rx) and bool(z_ok)
                    # z' should be 1 when keeping x is better than revised z on touched lines.
                    rzp = rx and (not bool(z_ok))
                else:
                    rz = False
                    rzp = rx
        cid = classify_line_case_id(
            rx=rx,
            ry=ry,
            ryp=ryp,
            rz=rz,
            rzp=rzp,
            touched=touched,
        )
        if 1 <= cid <= 6:
            counts[cid] = counts.get(cid, 0) + 1
        for d in line_details:
            if d.get("line") == lid:
                d["verifier_candidate_texts"] = ver_texts_line
                d["verifier_candidate_rewards"] = rs_line
                d["verifier_selected_idx"] = bi
                d["verifier_selected_json"] = raw_best or None
                d["signals"] = {
                    "rx": rx,
                    "ry": ry,
                    "ry_prime": ryp,
                    "rz": rz,
                    "rz_prime": rzp,
                    "touched": touched,
                    "case_id": cid,
                }
                if raw_best:
                    d["revision_text"] = rtext if "rtext" in locals() else None
                    d["revision_json"] = (
                        {
                            "form": z_line.form,
                            "line": z_line.line,
                            "description": z_line.description,
                            "amount": z_line.amount,
                            "rationale": z_line.rationale,
                        }
                        if "z_line" in locals() and z_line is not None
                        else None
                    )
                break

    r_f = torch.tensor(rewards_f_all, device=device, dtype=torch.float32)
    r_v = torch.tensor(rewards_v_all, device=device, dtype=torch.float32)
    adv_f = (r_f - r_f.mean()) / (r_f.std(unbiased=False) + 1e-8)
    adv_v = (r_v - r_v.mean()) / (r_v.std(unbiased=False) + 1e-8)
    return (
        seqs_f_all,
        plen_f_all,
        adv_f,
        r_f,
        seqs_v_all,
        plen_v_all,
        adv_v,
        r_v,
        draft_x,
        counts,
        line_details,
    )

def revise_filer_sample_for_taxonomy(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    draft_x: DraftReturn,
    verifier_raw: Dict[str, Any],
    gold_path: Path,
    args: argparse.Namespace,
    device: str,
    tc: CaseTaxonomyConfig,
) -> Tuple[float, bool, Optional[DraftReturn], str]:
    """One on-policy filer sample π(z | s, x, y) for case labels only (no GRPO backward)."""
    fb = build_revise_feedback_dict(verifier_raw, draft_x, sac_only=True)
    filer_user = json.dumps({"input": input_json, "feedback": fb}, indent=2)
    filer_text = build_chat_text(tokenizer, _ds_cfg().filer_system_prompt, filer_user)
    enc_f = tokenizer(
        filer_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    pid_f = enc_f["input_ids"].to(device)
    mask_f = enc_f["attention_mask"].to(device)
    prompt_len = int(pid_f.shape[1])
    seqs_f, _ = generate_group_sequences(
        model,
        tokenizer,
        pid_f,
        mask_f,
        1,
        args.filer_max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )
    text = tokenizer.decode(seqs_f[0][0, prompt_len:], skip_special_tokens=True)
    z = try_parse_draft(text)
    if z is None:
        return 0.0, False, None, text
    z_acc = float(reward_filer_draft(z, gold_path))
    keep = drafts_near_identical(
        draft_x, z, abs_tol=float(tc.revise_draft_amount_abs_tol)
    )
    return z_acc, keep, z, text
