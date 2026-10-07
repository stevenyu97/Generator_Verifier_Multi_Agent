"""DPA-GRPO: updates module."""
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

from dpa_grpo.checkpointing import _set_active_adapter
from dpa_grpo.config import _approver_line_prompt, _filer_line_prompt, _line_filer_user_payload, _line_revise_user_payload, _line_verifier_user_payload, _verifier_system_prompt_for
from dpa_grpo.losses import grpo_group_accumulate_gradients, ppo_clip_accumulate_gradients
from dpa_grpo.parsing import build_chat_text, _parse_approver_decision
from dpa_grpo.sampling import _build_seq_from_prompt_and_action, _sample_line_action, completion_token_logprobs
from benchmarks.tax.tax_line_case import classify_line_case_id, line_case_loss_weights



def _apply_transition_batch_update(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: AdamW,
    scaler: Optional[Any],
    transitions: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: str,
    use_amp: bool,
) -> Tuple[float, float]:
    if not transitions:
        return 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    lf_total = 0.0
    lv_total = 0.0
    n_backward = 0  # count of grpo_group_accumulate_gradients invocations
    skip_v_on_invalid = bool(
        getattr(args, "skip_verifier_update_on_invalid_filer", True)
    )
    skip_f_on_invalid_revise = bool(
        getattr(args, "skip_filer_revise_update_on_invalid_revise", True)
    )

    # ------------------------------------------------------------------
    # Approver step-level homeostasis + case-rarity weighting (anti-collapse).
    #
    # tenth_run revealed that the Approver collapses to "always KEEP" by step 6
    # when the case mix is dominated by case-1 (filer right, no SAC). The
    # cave-penalty alone is asymmetric and leaves "always KEEP" as a stable
    # local optimum. Two complementary terms, both opt-in (defaults disabled
    # to preserve prior behavior):
    #   (1) Homeostatic balance: when this step's observed Approver REVISE
    #       rate falls below --approver-target-revise-rate, add a uniform
    #       bonus to every revise_reward in the step. The bonus magnitude is
    #       balance_strength * max(target - observed, 0). Costs nothing when
    #       revise_rate >= target.
    #   (2) Case-rarity weighting: scale the per-transition Approver gradient
    #       by inverse-frequency of its case_id within the step batch, so the
    #       rare case-5a transitions (typically 1-2/step) contribute as much
    #       gradient as the abundant case-1 transitions.
    target_rev_rate = float(getattr(args, "approver_target_revise_rate", 0.30))
    balance_strength = float(getattr(args, "approver_balance_strength", 0.0))
    rare_alpha = float(getattr(args, "approver_rare_case_weight", 0.0))
    considered_trs = [
        t for t in transitions if bool((t.get("approver") or {}).get("considered", False))
    ]
    n_considered = len(considered_trs)
    if n_considered > 0 and balance_strength > 0:
        n_revise = sum(
            1 for t in considered_trs
            if str((t.get("approver") or {}).get("decision", "KEEP")).upper() == "REVISE"
        )
        revise_share = n_revise / float(n_considered)
        revise_bonus = balance_strength * max(target_rev_rate - revise_share, 0.0)
    else:
        revise_share = 0.0
        revise_bonus = 0.0
    if n_considered > 0 and rare_alpha > 0:
        case_counts: Dict[int, int] = {}
        for t in considered_trs:
            cid = int(t.get("case_id", 0))
            case_counts[cid] = case_counts.get(cid, 0) + 1
        n_distinct = max(len(case_counts), 1)
        avg_count = n_considered / float(n_distinct)
    else:
        case_counts = {}
        avg_count = 1.0
    # ------------------------------------------------------------------

    for tr in transitions:
        roles = tr["roles"]
        lid = tr["line_id"]
        filer_parsed = bool(tr.get("filer_parsed", True))
        revise_parsed = tr.get("revision", {}).get("parsed", None)

        # Filer-only baseline branch (--no-multi-agent). Run a paired GRPO
        # update over K=2 sampled drafts using oracle line correctness as
        # reward, then skip the Verifier/Filer-revise/Approver paths
        # entirely. This isolates "trained Filer alone" so Table 1 has a
        # clean control row for the multi-agent loop's contribution claim.
        if bool(tr.get("filer_only", False)):
            fp = tr.get("filer_pair") or {}
            samples = list(fp.get("samples") or [])
            rewards = list(fp.get("rewards") or [])
            f_user = fp.get("user", "")
            n_group = min(len(samples), len(rewards))
            if n_group >= 2 and f_user:
                _set_active_adapter(model, roles["filer_adapter"])
                seqs_f: List[torch.Tensor] = []
                pl_f: List[int] = []
                for sample_text in samples[:n_group]:
                    seq, pl = _build_seq_from_prompt_and_action(
                        tokenizer,
                        _filer_line_prompt(),
                        f_user,
                        sample_text,
                        device,
                    )
                    seqs_f.append(seq)
                    pl_f.append(pl)
                rfs = torch.tensor(
                    rewards[:n_group], device=device, dtype=torch.float32
                )
                # Skip update when every sample has the same reward: the
                # GRPO advantage is zero everywhere and the gradient is zero
                # too (advantage standardisation is undefined for std=0).
                # DAPO redraws tied groups before this point; a group that
                # is still tied after the cap is dropped here.
                if abs(float(rfs.std(unbiased=False).item())) > 1e-12:
                    adv_f = (rfs - rfs.mean()) / (
                        rfs.std(unbiased=False) + 1e-8
                    )
                    use_dapo = bool(getattr(args, "use_dapo", False))
                    use_gspo = bool(getattr(args, "use_gspo", False))
                    if use_dapo:
                        lf_total += ppo_clip_accumulate_gradients(
                            model,
                            seqs_f,
                            pl_f,
                            adv_f,
                            device,
                            use_amp,
                            scaler,
                            mode="dapo",
                            clip_low=float(getattr(args, "dapo_clip_low", 0.2)),
                            clip_high=float(getattr(args, "dapo_clip_high", 0.28)),
                            loss_scale=1.0,
                        )
                    elif use_gspo:
                        lf_total += ppo_clip_accumulate_gradients(
                            model,
                            seqs_f,
                            pl_f,
                            adv_f,
                            device,
                            use_amp,
                            scaler,
                            mode="gspo",
                            gspo_clip=float(getattr(args, "gspo_clip", 0.2)),
                            loss_scale=1.0,
                        )
                    else:
                        lf_total += grpo_group_accumulate_gradients(
                            model,
                            seqs_f,
                            pl_f,
                            adv_f,
                            args.beta_kl,
                            device,
                            use_amp,
                            scaler,
                            loss_scale=1.0,
                        )
                    n_backward += 1
            continue  # done with this transition; skip V/A/revise paths

        # Verifier counterfactual paired action loss. Train the SAC and NO_SAC stubs
        # every transition using the oracle label. Skip when the filer line was
        # invalid JSON: those rows carry an artificial sac_correct_line=True label,
        # which would otherwise keep rewarding SAC and collapse the verifier to
        # always-SAC.
        if not (skip_v_on_invalid and not filer_parsed):
            _set_active_adapter(model, roles["verifier_adapter"])
            v_user = tr["prompts"]["verifier_user"]
            v_sys_prompt = _verifier_system_prompt_for(args)
            v_loss_scale = float(getattr(args, "verifier_loss_scale", 1.0))
            v_actions = [
                json.dumps({"target_line": lid, "decision": "SAC"}, ensure_ascii=False),
                json.dumps({"target_line": lid, "decision": "NO_SAC"}, ensure_ascii=False),
            ]
            seqs_v: List[torch.Tensor] = []
            pl_v: List[int] = []
            for act in v_actions:
                seq, pl = _build_seq_from_prompt_and_action(
                    tokenizer, v_sys_prompt, v_user, act, device
                )
                seqs_v.append(seq)
                pl_v.append(pl)
            rvs = torch.tensor(
                [tr["verifier_rewards"]["SAC"], tr["verifier_rewards"]["NO_SAC"]],
                device=device,
                dtype=torch.float32,
            )
            adv_v = (rvs - rvs.mean()) / (rvs.std(unbiased=False) + 1e-8)
            lv_total += grpo_group_accumulate_gradients(
                model,
                seqs_v,
                pl_v,
                adv_v,
                args.beta_kl,
                device,
                use_amp,
                scaler,
                loss_scale=v_loss_scale,
            )
            n_backward += 1

        # Revision-text GRPO group: every parsed K-sample, scored by S_z.
        # Unparsed revisions are dropped. A group with no reward contrast
        # (all hits or all misses) is skipped.
        rev_samples = list((tr.get("revision") or {}).get("samples") or [])
        if skip_f_on_invalid_revise:
            rev_samples = [s for s in rev_samples if s.get("parsed") and s.get("text")]
        do_filer_revise_update = (
            bool(tr["revision"]["considered"]) and len(rev_samples) >= 2
        )
        if do_filer_revise_update:
            _set_active_adapter(model, roles["filer_adapter"])
            f_user = tr["prompts"]["revise_user"]
            seqs_r: List[torch.Tensor] = []
            pl_r: List[int] = []
            rewards_r: List[float] = []
            for sample in rev_samples:
                seq, pl = _build_seq_from_prompt_and_action(
                    tokenizer,
                    _filer_line_prompt(),
                    f_user,
                    str(sample["text"]),
                    device,
                )
                seqs_r.append(seq)
                pl_r.append(pl)
                rewards_r.append(float(sample.get("reward", 0.0)))
            rfs = torch.tensor(rewards_r, device=device, dtype=torch.float32)
            if abs(float(rfs.std(unbiased=False).item())) > 1e-12:
                adv_f = (rfs - rfs.mean()) / (rfs.std(unbiased=False) + 1e-8)
                lf_total += grpo_group_accumulate_gradients(
                    model,
                    seqs_r,
                    pl_r,
                    adv_f,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=1.0,
                )
                n_backward += 1

        # Approver KEEP/REVISE pair on the shown revision. Skipped when that
        # revision is not valid JSON, so the shared adapter is not updated
        # from a garbage revise prompt.
        ap = tr.get("approver", {}) or {}
        # No approver update when the shown revision is not valid JSON.
        if bool(ap.get("considered", False)) and revise_parsed is not False:
            _set_active_adapter(model, roles["approver_adapter"])
            a_user = tr["prompts"].get("approver_user", "") or ""
            if a_user:
                lid = tr["line_id"]
                ap_actions = [
                    json.dumps(
                        {"target_line": lid, "decision": "KEEP"}, ensure_ascii=False
                    ),
                    json.dumps(
                        {"target_line": lid, "decision": "REVISE"}, ensure_ascii=False
                    ),
                ]
                seqs_a: List[torch.Tensor] = []
                pl_a: List[int] = []
                for act in ap_actions:
                    seq, pl = _build_seq_from_prompt_and_action(
                        tokenizer, _approver_line_prompt(), a_user, act, device
                    )
                    seqs_a.append(seq)
                    pl_a.append(pl)
                # (B) Homeostatic balance bonus: shift REVISE reward up by the
                # step-level deficit. Computed once per step (above), applied
                # uniformly to every transition this step. Zero when disabled
                # (balance_strength=0) or when revise_rate >= target.
                ap_keep_r = float(ap.get("keep_reward", 0.0))
                ap_rev_r = float(ap.get("revise_reward", 0.0)) + revise_bonus
                ras = torch.tensor(
                    [ap_keep_r, ap_rev_r],
                    device=device,
                    dtype=torch.float32,
                )
                adv_a = (ras - ras.mean()) / (ras.std(unbiased=False) + 1e-8)
                # (C) Case-rarity weighting: scale this transition's gradient
                # by inverse frequency of its case_id in the step batch. Power
                # alpha controls strength: 0 disables (default), 0.5 = sqrt
                # damping, 1.0 = full inverse-frequency.
                if rare_alpha > 0 and case_counts:
                    cid = int(tr.get("case_id", 0))
                    cnt = float(case_counts.get(cid, 1))
                    rarity_scale = (avg_count / max(cnt, 1.0)) ** rare_alpha
                else:
                    rarity_scale = 1.0
                lf_total += grpo_group_accumulate_gradients(
                    model,
                    seqs_a,
                    pl_a,
                    adv_a,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=float(rarity_scale),
                )
                n_backward += 1
    # If every transition skipped every gradient path (e.g. an entire case with
    # zero SAC raises and zero invalid filer parses ─ the verifier still runs ─
    # OR an entire case with all filer JSONs invalid AND no SAC raises), there
    # are no gradients to step. Calling scaler.step() in that case raises
    # AssertionError("No inf checks were recorded for this optimizer"). Skip
    # cleanly instead.
    if n_backward == 0:
        return 0.0, 0.0
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return lf_total, lv_total
