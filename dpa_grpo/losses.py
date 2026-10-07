"""DPA-GRPO: losses module."""
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

from dpa_grpo.sampling import (
    completion_logprob_sum,
    completion_token_logprobs,
    ref_context_manager,
)

def grpo_group_accumulate_gradients(
    model: torch.nn.Module,
    sequences: List[torch.Tensor],
    prompt_lengths: List[int],
    advantages: torch.Tensor,
    beta_kl: float,
    device: str,
    use_amp: bool,
    scaler: Optional[Any],
    loss_scale: float = 1.0,
    *,
    length_normalize: bool = True,
) -> float:
    """One GRPO group: backward **per sequence** so peak memory is O(1) seq graph, not O(K).

    Gradients accumulate to match the mean of per-sequence GRPO losses. Returns that mean
    (detached) for logging.

    With ``length_normalize`` the policy term uses the **mean** per-token log-prob rather
    than the sum. A summed log-prob makes the gradient magnitude proportional to completion
    length, so for a paired binary decision the branch that emits more text dominates the
    update even when both branches carry the same reward. In the dual setting a SAC verdict
    is a long structured JSON while NO_SAC is a short refusal (~3x on tax), which ratchets
    the intervention rate toward 1.0 and then compounds via the revision branch. Set to
    False only to reproduce pre-fix runs.
    """
    ctx = ref_context_manager(model)
    model.train()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_on = bool(use_amp and device_type == "cuda")
    k = max(len(sequences), 1)
    total_log = 0.0
    with torch.amp.autocast(
        device_type=device_type,
        enabled=amp_on,
        dtype=torch.bfloat16,
    ):
        for i, (seq, plen) in enumerate(zip(sequences, prompt_lengths)):
            adv_f = float(advantages[i].item())
            if length_normalize:
                tok_lp = completion_token_logprobs(
                    model, seq, plen, enable_grad=True
                )
                if tok_lp.numel() == 0:
                    continue
                n_tok = int(tok_lp.numel())
                if ctx is not None:
                    with ctx():
                        ref_tok = completion_token_logprobs(
                            model, seq, plen, enable_grad=False
                        )
                    n_tok = int(min(n_tok, ref_tok.numel())) or n_tok
                    ref_val = (
                        float(ref_tok[:n_tok].mean()) if ref_tok.numel() else None
                    )
                else:
                    ref_val = None
                lp = tok_lp[:n_tok].mean()
                ref_lp = lp.detach() if ref_val is None else ref_val
            else:
                lp = completion_logprob_sum(model, seq, plen, enable_grad=True)
                if ctx is not None:
                    with ctx():
                        ref_lp = completion_logprob_sum(
                            model, seq, plen, enable_grad=False
                        )
                else:
                    ref_lp = lp.detach()
            loss_term = -adv_f * lp + beta_kl * (lp - ref_lp)
            total_log += float(loss_term.detach().cpu()) / k * loss_scale
            to_backward = loss_term / k * float(loss_scale)
            if scaler is not None:
                scaler.scale(to_backward).backward()
            else:
                to_backward.backward()
    # Avoid torch.cuda.empty_cache() here: it synchronizes and often reports async CUDA
    # errors at this line even when the failing kernel ran earlier.
    return total_log

def ppo_clip_accumulate_gradients(
    model: torch.nn.Module,
    sequences: List[torch.Tensor],
    prompt_lengths: List[int],
    advantages: torch.Tensor,
    device: str,
    use_amp: bool,
    scaler: Optional[Any],
    *,
    mode: str = "dapo",
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    gspo_clip: float = 0.2,
    loss_scale: float = 1.0,
) -> float:
    """Unified PPO-clip group update for DAPO and GSPO baselines.

    The two modes share the importance-ratio scaffolding (capture ``π_old``
    once per sequence with ``enable_grad=False``, recompute ``π_new`` with
    grad, form ``log_ratio = lp_new - lp_old.detach()``) and differ only in
    the clip shape and the reduction:

    - ``mode='dapo'`` — token-level reduction with **asymmetric** clip:
      per-token ratio ``ρ_t = exp(log_ratio_t)``, surrogate
      ``min(ρ_t · A, clip(ρ_t, 1 − ε_low, 1 + ε_high) · A)``, then average
      over completion tokens within each sequence (matches Eq. 8 of the
      DAPO paper).
    - ``mode='gspo'`` — sequence-level reduction with **symmetric** clip:
      sequence ratio ``ρ_seq = exp(mean_t(log_ratio_t))`` (geometric mean
      of per-token ratios), surrogate
      ``min(ρ_seq · A, clip(ρ_seq, 1 − ε, 1 + ε) · A)``, taken once per
      sequence (matches the GSPO formulation in the Qwen team's paper).

    Backward is per-sequence so peak memory is O(1) sequence-graph rather
    than O(K) like a fused batch-backward would be — same pattern as
    ``grpo_group_accumulate_gradients``. The KL anchor against the
    reference policy is intentionally **dropped** (β_KL = 0) because both
    DAPO and GSPO rely on the importance-ratio clip alone to control
    policy drift; reintroducing the GRPO KL term here would muddy the
    comparison. Dynamic-sampling filtering (skip groups with all-equal
    rewards) is the caller's responsibility — it is already enforced by
    the existing ``std > 1e-12`` guard before this function is called.

    Returns the mean loss (detached, scalar) for logging.
    """
    if mode not in ("dapo", "gspo"):
        raise ValueError(f"unknown ppo-clip mode: {mode}")
    model.train()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_on = bool(use_amp and device_type == "cuda")
    k = max(len(sequences), 1)
    total_log = 0.0
    with torch.amp.autocast(
        device_type=device_type,
        enabled=amp_on,
        dtype=torch.bfloat16,
    ):
        for i, (seq, plen) in enumerate(zip(sequences, prompt_lengths)):
            adv_f = float(advantages[i].item())
            # π_old: rollout-time policy. Within a single training step the
            # optimizer hasn't been stepped yet (paired-action GRPO accumulates
            # gradients across transitions and steps once at the end), so the
            # current model parameters are still the rollout-time parameters
            # at the time we hit this line on the first transition. Across
            # transitions within the same step the parameters remain unchanged
            # too (no inner step), so this lp_old is faithful to PPO's
            # "pre-update policy" semantics for this on-policy setup.
            lp_old = completion_token_logprobs(
                model, seq, plen, enable_grad=False
            ).detach()
            lp_new = completion_token_logprobs(
                model, seq, plen, enable_grad=True
            )
            if lp_old.numel() == 0 or lp_new.numel() == 0:
                continue
            # Tensor shape mismatch shouldn't happen (same input + slice) but
            # be defensive: take the common prefix.
            n = int(min(lp_old.numel(), lp_new.numel()))
            lp_old = lp_old[:n]
            lp_new = lp_new[:n]
            log_ratio = lp_new - lp_old

            if mode == "dapo":
                # Token-level: per-token ratios, asymmetric clip, mean over
                # completion tokens. min(surr_unclipped, surr_clipped) is the
                # PPO-clip pessimistic bound; with positive advantage it caps
                # the upside and with negative advantage it caps the downside.
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_f
                surr2 = torch.clamp(
                    ratio,
                    1.0 - float(clip_low),
                    1.0 + float(clip_high),
                ) * adv_f
                per_token_obj = torch.minimum(surr1, surr2)
                loss_seq = -per_token_obj.mean()
            else:  # gspo
                # Sequence-level: geometric mean of token ratios, symmetric
                # clip. exp(mean(log_ratio)) is numerically more stable than
                # prod(exp(log_ratio))**(1/n) for long sequences.
                seq_log_ratio = log_ratio.mean()
                ratio_seq = torch.exp(seq_log_ratio)
                surr1 = ratio_seq * adv_f
                surr2 = torch.clamp(
                    ratio_seq,
                    1.0 - float(gspo_clip),
                    1.0 + float(gspo_clip),
                ) * adv_f
                loss_seq = -torch.minimum(surr1, surr2)

            total_log += float(loss_seq.detach().cpu()) / k * loss_scale
            to_backward = loss_seq / k * float(loss_scale)
            if scaler is not None:
                scaler.scale(to_backward).backward()
            else:
                to_backward.backward()
    return total_log
