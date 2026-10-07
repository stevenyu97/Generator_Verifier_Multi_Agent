"""DPA-GRPO: dual_rollout module."""
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

import random
from core.agents import parse_verifier_raw_to_safety_case
from core.client import extract_json_from_response
from dpa_grpo.checkpointing import _set_active_adapter
from dpa_grpo.config import (
    _approver_line_prompt,
    _ds_cfg,
    _filer_line_prompt,
    _gold_path,
    _line_approver_user_payload,
    _line_filer_user_payload,
    _line_revise_user_payload,
    _line_verifier_user_payload,
    _verifier_system_prompt_for,
)
from dpa_grpo.parsing import try_parse_line_draft, _parse_approver_decision, _extract_line_verdict_from_raw
from dpa_grpo.sampling import _role_assignment, _sample_line_action
from benchmarks.line_eval import evaluated_line_ids, line_strict_correctness_by_line_id
from core.rewards import reward_filer_draft
from core.schemas import DraftLine, DraftReturn



def _dual_linewise_rollout_transitions(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    gold_path: Path,
    args: argparse.Namespace,
    device: str,
    iter_id: int,
) -> Tuple[List[Dict[str, Any]], DraftReturn, DraftReturn]:
    roles = _role_assignment(
        iter_id,
        args.role_swap_period,
        swap_enabled=bool(getattr(args, "swap_roles", False)),
    )
    filer_only_mode = bool(getattr(args, "no_multi_agent", False))
    line_ids = list(evaluated_line_ids(gold_path))
    context_so_far: List[Dict[str, Any]] = []
    selected_lines: List[DraftLine] = []
    post_revise_lines: List[DraftLine] = []
    transitions: List[Dict[str, Any]] = []

    for lid in line_ids:
        _set_active_adapter(model, roles["filer_adapter"])
        f_user = _line_filer_user_payload(input_json, lid, context_so_far)

        if filer_only_mode:
            # Filer-only baseline: sample K drafts (args.group_size), reward
            # each by oracle line correctness S_x, and train the Filer adapter
            # on that group. Evaluation draws K=1. DAPO redraws a tied group
            # until the rewards disagree or the resample cap is hit. GRPO and
            # GSPO keep the first group and the update skips it when every
            # reward is equal. The first highest-reward draft is used for
            # context_so_far.
            group_k = max(int(getattr(args, "group_size", 2) or 2), 2)
            if not model.training:
                group_k = 1
            use_dapo_sample = bool(getattr(args, "use_dapo", False)) and model.training
            resample_cap = (
                max(int(getattr(args, "dapo_resample_cap", 4) or 0), 0)
                if use_dapo_sample
                else 0
            )

            def _sample_filer_group() -> Tuple[List[str], List[Optional[DraftLine]], List[float]]:
                texts: List[str] = []
                lines: List[Optional[DraftLine]] = []
                rewards: List[float] = []
                for _k in range(group_k):
                    cand_text = _sample_line_action(
                        model,
                        tokenizer,
                        _filer_line_prompt(),
                        f_user,
                        args.filer_max_new_tokens,
                        args,
                        device,
                    )
                    cand_line = try_parse_line_draft(cand_text, lid)
                    if cand_line is None:
                        cand_S = 0.0
                    else:
                        cand_tmp = DraftReturn(
                            return_version=_ds_cfg().return_version, lines=[cand_line], metadata={}
                        )
                        cand_S = (
                            1.0
                            if line_strict_correctness_by_line_id(cand_tmp, gold_path).get(
                                lid, False
                            )
                            else 0.0
                        )
                    texts.append(cand_text)
                    lines.append(cand_line)
                    rewards.append(cand_S)
                return texts, lines, rewards

            pair_texts, pair_lines, pair_S = _sample_filer_group()
            draws = 1
            while (
                draws <= resample_cap
                and pair_S
                and (max(pair_S) - min(pair_S)) <= 1e-8
            ):
                pair_texts, pair_lines, pair_S = _sample_filer_group()
                draws += 1
            best_idx = 0
            for i, reward in enumerate(pair_S):
                if reward > pair_S[best_idx]:
                    best_idx = i
            x_parsed = pair_lines[best_idx]
            f_text = pair_texts[best_idx]
            filer_parsed = x_parsed is not None
            if x_parsed is None:
                x_line = DraftLine(
                    form=_ds_cfg().form_name,
                    line=lid,
                    description=f"Line {lid}",
                    amount=0.0,
                    rationale="invalid_json",
                )
            else:
                x_line = x_parsed
            selected_lines.append(x_line)
            post_revise_lines.append(x_line)  # no revision in filer-only
            context_so_far.append(
                {
                    "form": x_line.form,
                    "line": x_line.line,
                    "description": x_line.description,
                    "amount": x_line.amount,
                    "rationale": x_line.rationale,
                }
            )
            S_x = pair_S[best_idx]
            t_x_line = int(S_x > 0.5)
            transitions.append(
                {
                    "iter_id": iter_id,
                    "s_id": str(gold_path.parent.resolve()),
                    "line_id": lid,
                    "roles": roles,
                    "filer_only": True,
                    "filer_pair": {
                        "user": f_user,
                        "samples": pair_texts,
                        "rewards": pair_S,
                    },
                    "filer_line_raw": f_text,
                    "filer_parsed": bool(filer_parsed),
                    "filer_line_struct": {
                        "form": x_line.form,
                        "line": x_line.line,
                        "description": x_line.description,
                        "amount": x_line.amount,
                        "rationale": x_line.rationale,
                    },
                    "y_choice": "NO_SAC",
                    "y_text": "",
                    "oracle": {
                        "t_x_line": t_x_line,
                        "S_x_line": S_x,
                        "S_z_line": S_x,
                        "delta_line": 0.0,
                        "sac_correct_line": False,
                    },
                    "verifier_rewards": {"SAC": 0.0, "NO_SAC": 0.0},
                    "revision": {
                        "considered": False,
                        "parsed": None,
                        "keep_text": "",
                        "revise_text": "",
                        "keep_reward": S_x,
                        "revise_reward": S_x,
                        "decision": "KEEP",
                        "outcome_bonus": 0.0,
                    },
                    "approver": {
                        "considered": False,
                        "parsed": None,
                        "text": "",
                        "decision": "KEEP",
                        "keep_reward": 0.0,
                        "revise_reward": 0.0,
                        "sac_was_correct": False,
                        "decision_was_right": None,
                    },
                    "prompts": {
                        "filer_user": f_user,
                        "verifier_user": "",
                        "revise_user": "",
                        "approver_user": "",
                    },
                    "case_sub": "filer_only",
                    "case_id": 1 if S_x > 0.5 else 4,
                }
            )
            continue  # next line; skip Verifier+Approver path entirely

        f_text = _sample_line_action(
            model,
            tokenizer,
            _filer_line_prompt(),
            f_user,
            args.filer_max_new_tokens,
            args,
            device,
        )
        x_parsed = try_parse_line_draft(f_text, lid)
        filer_parsed = x_parsed is not None
        if x_parsed is None:
            x_line = DraftLine(
                form=_ds_cfg().form_name,
                line=lid,
                description=f"Line {lid}",
                amount=0.0,
                rationale="invalid_json",
            )
        else:
            x_line = x_parsed
        selected_lines.append(x_line)
        context_so_far.append(
            {
                "form": x_line.form,
                "line": x_line.line,
                "description": x_line.description,
                "amount": x_line.amount,
                "rationale": x_line.rationale,
            }
        )
        x_tmp = DraftReturn(return_version=_ds_cfg().return_version, lines=[x_line], metadata={})
        rx = bool(line_strict_correctness_by_line_id(x_tmp, gold_path).get(lid, False))
        S_x = 1.0 if rx else 0.0
        t_x_line = int(rx)
        sac_correct_line = not rx

        _set_active_adapter(model, roles["verifier_adapter"])
        v_user = _line_verifier_user_payload(input_json, x_line, lid, context_so_far[:-1])
        v_sys_prompt = _verifier_system_prompt_for(args)
        y_text = _sample_line_action(
            model,
            tokenizer,
            v_sys_prompt,
            v_user,
            args.verifier_max_new_tokens,
            args,
            device,
        )
        y_raw = extract_json_from_response(y_text.strip()) or {}
        verdict = _extract_line_verdict_from_raw(y_raw, lid)
        y_choice = "SAC" if verdict != "correct" else "NO_SAC"
        # Counterfactual SAC/NO_SAC reward: train both action stubs every transition
        # using the oracle label, regardless of which one the verifier actually emitted.
        verifier_pair_rewards = {
            "SAC": 1.0 if sac_correct_line else 0.0,
            "NO_SAC": 0.0 if sac_correct_line else 1.0,
        }

        revision_considered = y_choice == "SAC"
        revise_text = ""
        revise_samples: List[Dict[str, Any]] = []
        z_line: Optional[DraftLine] = None
        S_z = S_x
        outcome_bonus = float(getattr(args, "filer_outcome_bonus", 0.5))
        fmt_pen = float(getattr(args, "revise_format_penalty", 0.25))
        keep_reward = S_x
        revise_reward = S_x
        approver_keep_reward = 0.0
        approver_revise_reward = 0.0
        revision_decision = "KEEP"
        approver_text = ""
        approver_parsed: Optional[bool] = None  # None when not considered
        a_user = ""
        revise_parsed: Optional[bool] = None  # None when not considered
        forced_explore = False  # set inside the revision_considered branch when
        # --approver-explore-eps fires; default False so the unconditional
        # transition record below always has a value.
        if revision_considered:
            _set_active_adapter(model, roles["filer_adapter"])
            r_user = _line_revise_user_payload(input_json, x_line, lid, y_raw, context_so_far[:-1])
            # Sample K revisions from the revise policy. Gold scores every
            # sample for the GRPO group; it does not choose which string the
            # approver sees. Evaluation draws K=1.
            revise_k = max(int(getattr(args, "revise_sample_k", 1) or 1), 1)
            if not model.training:
                revise_k = 1
            revise_temp = float(getattr(args, "revise_temperature", args.temperature))
            revise_samples: List[Dict[str, Any]] = []
            for _k in range(revise_k):
                cand_text = _sample_line_action(
                    model,
                    tokenizer,
                    _filer_line_prompt(),
                    r_user,
                    args.filer_max_new_tokens,
                    args,
                    device,
                    temperature=revise_temp,
                )
                cand_line = try_parse_line_draft(cand_text, lid)
                if cand_line is None:
                    cand_s = 0.0
                    cand_reward = -fmt_pen
                    cand_parsed = False
                else:
                    cand_tmp = DraftReturn(
                        return_version=_ds_cfg().return_version, lines=[cand_line], metadata={}
                    )
                    cand_s = (
                        1.0
                        if line_strict_correctness_by_line_id(cand_tmp, gold_path).get(
                            lid, False
                        )
                        else 0.0
                    )
                    cand_reward = cand_s
                    cand_parsed = True
                revise_samples.append(
                    {
                        "text": cand_text,
                        "parsed": cand_parsed,
                        "S_z": cand_s,
                        "reward": cand_reward,
                        "line": cand_line,
                    }
                )
            # First sample is the on-policy revision shown to the approver.
            shown = revise_samples[0]
            revise_text = str(shown["text"])
            z_line = shown["line"]
            revise_parsed = bool(shown["parsed"])
            S_z = float(shown["S_z"])
            keep_reward = float(S_x)
            revise_reward = float(shown["reward"])

            # Approver step: pinned-adapter sample of an explicit KEEP/REVISE
            # decision token, conditioned on (input, draft_line, verifier_feedback,
            # revised_line). The Approver lives on roles["approver_adapter"]
            # (always "default") so its policy survives role swaps.
            _set_active_adapter(model, roles["approver_adapter"])
            a_user = _line_approver_user_payload(
                input_json, x_line, lid, y_raw, z_line, context_so_far[:-1]
            )
            approver_text = _sample_line_action(
                model,
                tokenizer,
                _approver_line_prompt(),
                a_user,
                256,
                args,
                device,
            )
            parsed_decision = _parse_approver_decision(approver_text)
            approver_parsed = parsed_decision is not None
            revision_decision = parsed_decision if parsed_decision is not None else "KEEP"

            # Forced ε-exploration on the Approver decision (training rollouts
            # only). Once the Approver collapses to always-KEEP (~step 5 in
            # every multi-agent run we've measured), case-5a/5b transitions
            # vanish from case_counts, the rare-case-weight no longer applies
            # to them, and the gradient signal on "REVISE was right" rounds
            # away to nothing. With probability eps, override the sampled
            # decision to REVISE so those transitions resurface in the
            # case_histogram and feed the rare-case-weight pathway. The
            # paired-action GRPO loss is computed against canonical KEEP /
            # REVISE action strings regardless of the rolled-out action, so
            # forcing the rollout never produces an off-policy gradient.
            ap_eps = float(getattr(args, "approver_explore_eps", 0.0) or 0.0)
            if model.training and ap_eps > 0.0 and revision_decision != "REVISE":
                if random.random() < ap_eps:
                    revision_decision = "REVISE"
                    forced_explore = True

            # Counterfactual approver reward. The right meta-action depends
            # on whether the SAC's *suggestion* is good (operationalised as
            # whether the revised line lands on gold, S_z), not just on
            # whether SAC raising was justified. The Approver's job is the
            # judgment "accept good suggestions, reject bad ones":
            #
            #   - Filer correct, V SAC (false positive): KEEP is right, REVISE is wrong.
            #   - Filer wrong, V SAC, S_z=1 (good suggestion): REVISE is right (case 5a),
            #     KEEP misses the easy fix (case 6b).
            #   - Filer wrong, V SAC, S_z=0 (bad suggestion):  KEEP is right — A
            #     correctly rejects a bad SAC (case 6a); REVISE caves to a bad SAC
            #     and doesn't fix the line either (case 5b).
            # Sharper penalty for "Approver caves on a correct draft" failures.
            # Sixth_run case 1 line 1a was Sx=9/9 but the Verifier raised SAC 8/9
            # times and the Approver REVISEd 4/9 times anyway with Sz=5/9 — this
            # alone took post_revise from 9/9 to 5/9 for that line. A flat 0
            # revise_reward isn't a strong enough signal; we apply an explicit
            # negative reward when the Approver REVISEs a correct draft and the
            # revise sample also fails to land gold.
            cave_penalty = float(getattr(args, "approver_cave_penalty", 0.5))
            miss_penalty = float(getattr(args, "approver_miss_penalty", 0.0))
            if z_line is None:
                # No usable revision exists → KEEP is the right call regardless
                # (treat as a "bad SAC" sub-case for filer-wrong; A rightly KEEPs).
                approver_keep_reward = 1.0
                approver_revise_reward = 0.0 - fmt_pen
            elif not sac_correct_line:
                # Cases 2/3: filer was correct; SAC was a false positive.
                approver_keep_reward = 1.0 + outcome_bonus * float(S_x)
                if S_z < 0.5:
                    # Case 2 + bad revise: A caved AND the revise broke the line.
                    # Strict negative signal so the gap KEEP - REVISE is widened
                    # from ~1.5 to ~1.5 + cave_penalty.
                    approver_revise_reward = -cave_penalty
                else:
                    # Case 2 + revise also lands gold: A's call was still wrong
                    # (KEEP was right since draft was correct) but no harm done.
                    approver_revise_reward = 0.0 + outcome_bonus * float(S_z)
            else:
                # Cases 5/6: filer was wrong. Right call depends on S_z.
                if S_z > 0.5:
                    # SAC suggestion is good → REVISE is right (5a vs 6b).
                    # Apply the missed-opportunity penalty to KEEP: this is the
                    # case-6b "Approver missed an easy fix" failure mode that
                    # drives the always-KEEP collapse seen in 4b_5/4b_10/8b_3.
                    # Symmetric counterpart to the cave penalty above (which
                    # punishes case-2 false-positive REVISE on a correct draft).
                    approver_keep_reward = 0.0 - miss_penalty
                    approver_revise_reward = 1.0 + outcome_bonus * float(S_z)
                else:
                    # SAC suggestion is bad → KEEP is right (6a vs 5b).
                    approver_keep_reward = 1.0
                    approver_revise_reward = 0.0 + outcome_bonus * float(S_z)
        delta_line = S_z - S_x

        # Build the post-revise (approver-decided) line for a separate accuracy
        # measurement. We DO NOT thread this back into context_so_far — the
        # filer's later lines still see the original draft, mirroring how the
        # rollout is sampled. This is purely a measurement of "what the final
        # draft would look like if we honored every approver decision".
        if revision_considered and revision_decision == "REVISE" and z_line is not None:
            post_revise_lines.append(z_line)
        else:
            post_revise_lines.append(x_line)

        # 8-bucket linewise taxonomy {1, 2, 3, 4, 5a, 5b, 6a, 6b}. Each
        # transition lands in exactly one bucket (mutually exclusive
        # partition). Cases 2/3 are filer-correct; cases 5/6 are filer-wrong.
        # The a/b sub-split inside cases 5 and 6 reflects the SAC's
        # *suggestion quality* (is the revise sample actually good?), not the
        # SAC-raising decision itself — when filer is wrong, the SAC raise
        # is justified, but the verifier's specific suggestion may still be
        # right or wrong, and the Approver's judgment is what's measured.
        #
        #   1:  filer right, V NO_SAC                                 — all good
        #   2:  filer right, V SAC, A REVISE                          — V false pos + A caves
        #   3:  filer right, V SAC, A KEEP                            — V false pos, A right
        #   4:  filer wrong, V NO_SAC                                 — V missed bug
        #   5a: filer wrong, V SAC, A REVISE, S_z=1 (suggestion good) — A right (ideal)
        #   5b: filer wrong, V SAC, A REVISE, S_z=0 (suggestion bad)  — A wrong (caved to bad SAC)
        #   6a: filer wrong, V SAC, A KEEP,   S_z=0 (suggestion bad)  — A right (rejected bad SAC)
        #   6b: filer wrong, V SAC, A KEEP,   S_z=1 (suggestion good) — A wrong (missed easy fix)
        sac_was_correct = bool(sac_correct_line)  # True iff filer was wrong
        rx = bool(S_x > 0.5)
        rz = bool(S_z > 0.5)  # True iff revise sample lands gold (= SAC suggestion good)
        case_id: int = 0
        case_sub = "other"
        decision_was_right: Optional[bool] = None
        if not revision_considered:
            case_id = 1 if rx else 4
        elif rx:
            case_id = 2 if revision_decision == "REVISE" else 3
            decision_was_right = revision_decision == "KEEP"  # KEEP is right vs false pos
        else:
            # Filer wrong: SAC was rightly raised, but the suggestion's
            # quality (rz) determines whether A's call is right.
            if revision_decision == "REVISE":
                case_id = 5
                case_sub = "5a" if rz else "5b"
                decision_was_right = rz  # 5a right, 5b wrong
            else:
                case_id = 6
                case_sub = "6a" if not rz else "6b"
                decision_was_right = not rz  # 6a right, 6b wrong

        transitions.append(
            {
                "iter_id": iter_id,
                "s_id": str(gold_path.parent.resolve()),
                "line_id": lid,
                "roles": roles,
                "filer_line_raw": f_text,
                "filer_parsed": bool(filer_parsed),
                "filer_line_struct": {
                    "form": x_line.form,
                    "line": x_line.line,
                    "description": x_line.description,
                    "amount": x_line.amount,
                    "rationale": x_line.rationale,
                },
                "y_choice": y_choice,
                "y_text": y_text,
                "oracle": {
                    "t_x_line": t_x_line,
                    "S_x_line": S_x,
                    "S_z_line": S_z,
                    "delta_line": delta_line,
                    "sac_correct_line": bool(sac_correct_line),
                },
                "verifier_rewards": verifier_pair_rewards,
                "revision": {
                    "considered": revision_considered,
                    "parsed": revise_parsed,
                    "keep_text": json.dumps(
                        {
                            "form": x_line.form,
                            "line": x_line.line,
                            "description": x_line.description,
                            "amount": x_line.amount,
                            "rationale": x_line.rationale,
                        },
                        ensure_ascii=False,
                    ),
                    "revise_text": revise_text,
                    "samples": [
                        {
                            "text": s["text"],
                            "parsed": s["parsed"],
                            "S_z": s["S_z"],
                            "reward": s["reward"],
                        }
                        for s in revise_samples
                    ],
                    "keep_reward": keep_reward,
                    "revise_reward": revise_reward,
                    "decision": revision_decision,
                    "outcome_bonus": outcome_bonus,
                },
                "approver": {
                    "considered": revision_considered,
                    "parsed": approver_parsed,
                    "text": approver_text,
                    "decision": revision_decision,
                    "keep_reward": approver_keep_reward,
                    "revise_reward": approver_revise_reward,
                    # SAC-correctness signal (latent oracle): True iff the
                    # verifier *should* have raised SAC, i.e. the filer was
                    # wrong on this line. Distinguishes 5a vs 5b and 6a vs 6b.
                    "sac_was_correct": sac_was_correct,
                    "decision_was_right": decision_was_right,
                    # True iff the rolled-out decision was overridden to
                    # REVISE by --approver-explore-eps (forced exploration).
                    # Used by the step aggregator to log
                    # approver_forced_revise_rate for diagnostics.
                    "forced_explore": forced_explore,
                },
                "prompts": {
                    "filer_user": f_user,
                    "verifier_user": v_user,
                    "revise_user": r_user if revision_considered else "",
                    "approver_user": a_user,
                },
                "case_sub": case_sub,
                "case_id": int(case_id),
            }
        )

    draft_x = DraftReturn(return_version=_ds_cfg().return_version, lines=selected_lines, metadata={})
    draft_x_post_revise = DraftReturn(
        return_version=_ds_cfg().return_version, lines=post_revise_lines, metadata={}
    )
    return transitions, draft_x, draft_x_post_revise
