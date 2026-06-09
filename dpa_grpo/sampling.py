"""DPA-GRPO: sampling module."""
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

from dpa_grpo.parsing import build_chat_text



def completion_logprob_sum(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_length: int,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Sum of log p(token) over generated continuation (first gen index ``prompt_length``).

    Use ``enable_grad=False`` for the reference-policy forward (adapter disabled): avoids an
    extra autograd graph and reduces bad interactions with checkpointing / LoRA on some setups.
    """
    if input_ids.dim() != 2:
        raise ValueError("expected input_ids [1, L]")
    input_ids = input_ids.long()
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    seq_len = int(input_ids.size(1))
    if max_pos is not None and seq_len > max_pos:
        raise ValueError(
            f"Sequence length {seq_len} exceeds model max_position_embeddings ({max_pos}); "
            "lower --filer-max-new-tokens / --verifier-max-new-tokens or shorten prompts."
        )
    ctx_grad = torch.enable_grad() if enable_grad else torch.inference_mode()
    with ctx_grad:
        out = model(input_ids)
        logits = out.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        vocab = logits.size(-1)
        ti = targets.clamp(min=0, max=vocab - 1)
        token_lp = log_probs.gather(-1, ti.unsqueeze(-1)).squeeze(-1)
        if prompt_length <= 0 or prompt_length >= input_ids.size(1):
            return token_lp.sum() * 0.0
        start = prompt_length - 1
        seg = token_lp[:, start:]
        return seg.sum()

def completion_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_length: int,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Per-token log-probs over the completion segment.

    Returns a 1-D tensor of shape ``[num_completion_tokens]``. Differs from
    ``completion_logprob_sum`` in that we keep the per-token resolution that
    PPO-clip variants (DAPO token-level loss, GSPO sequence-level ratio)
    require to form per-token / per-sequence importance ratios. Set
    ``enable_grad=False`` for the rollout-time ``π_old`` capture.
    """
    if input_ids.dim() != 2:
        raise ValueError("expected input_ids [1, L]")
    input_ids = input_ids.long()
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    seq_len = int(input_ids.size(1))
    if max_pos is not None and seq_len > max_pos:
        raise ValueError(
            f"Sequence length {seq_len} exceeds model max_position_embeddings ({max_pos})"
        )
    ctx_grad = torch.enable_grad() if enable_grad else torch.inference_mode()
    with ctx_grad:
        out = model(input_ids)
        logits = out.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        vocab = logits.size(-1)
        ti = targets.clamp(min=0, max=vocab - 1)
        token_lp = log_probs.gather(-1, ti.unsqueeze(-1)).squeeze(-1)
        if prompt_length <= 0 or prompt_length >= input_ids.size(1):
            return token_lp.new_zeros(0)
        start = prompt_length - 1
        return token_lp[0, start:]

def ref_context_manager(model: torch.nn.Module) -> Optional[Callable[[], Any]]:
    if _HAS_PEFT and PeftModel is not None and isinstance(model, PeftModel):
        return model.disable_adapter
    return None

def generate_group_sequences(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    model.eval()
    prompt_len = int(prompt_ids.shape[1])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bad_words_ids: List[List[int]] = []
    try:
        for w in ("<think>", "</think>", "<tool_response>", "</tool_response>"):
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)
    except Exception:
        bad_words_ids = []
    out_list: List[torch.Tensor] = []
    lengths: List[int] = []
    with torch.no_grad():
        for _ in range(group_size):
            gen = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=pad_id,
                bad_words_ids=bad_words_ids or None,
            )
            out_list.append(gen)
            lengths.append(prompt_len)
    return out_list, lengths

def _build_seq_from_prompt_and_action(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    action_text: str,
    device: str,
) -> Tuple[torch.Tensor, int]:
    prompt_text = build_chat_text(tokenizer, system_prompt, user_text)
    p = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(p["input_ids"].shape[1])
    full_text = prompt_text + action_text
    f = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    return f["input_ids"].to(device), prompt_len

def _role_assignment(
    iter_id: int, period: int = 10, swap_enabled: bool = False
) -> Dict[str, str]:
    # The Approver is a head of the *Filer* policy in our two-player
    # framing (Filer vs. Verifier). It always shares the Filer's adapter
    # ("default" by default, "theta" only during the swapped block when
    # --swap-roles is set) so that the KEEP/REVISE decision is part of the
    # same player whose draft and revise policies are being trained.
    #
    # Role swapping for the Filer/Verifier pair is OFF by default and only
    # activates when --swap-roles is set. Without swapping, the Filer always
    # trains on the "default" adapter and the Verifier on "theta" (block 0
    # assignment). fifth_run showed that swapping disrupts learning for both
    # the Approver and the Verifier, so the fixed assignment is now the
    # baseline and swapping is opt-in for explicit symmetry experiments.
    if not swap_enabled:
        return {
            "A_role": "filer",
            "B_role": "verifier",
            "filer_adapter": "default",
            "verifier_adapter": "theta",
            "approver_adapter": "default",
            "swap_enabled": "false",
        }
    block = ((iter_id - 1) // max(period, 1)) % 2
    if block == 0:
        return {
            "A_role": "filer",
            "B_role": "verifier",
            "filer_adapter": "default",
            "verifier_adapter": "theta",
            "approver_adapter": "default",
            "swap_enabled": "true",
        }
    return {
        "A_role": "verifier",
        "B_role": "filer",
        "filer_adapter": "theta",
        "verifier_adapter": "default",
        "approver_adapter": "default",
        "swap_enabled": "true",
    }

def _sample_line_action(
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    max_new_tokens: int,
    args: argparse.Namespace,
    device: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    txt = build_chat_text(tokenizer, system_prompt, user_text)
    enc = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
    pid = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    plen = int(pid.shape[1])
    seqs, _ = generate_group_sequences(
        model,
        tokenizer,
        pid,
        mask,
        1,
        max_new_tokens,
        float(args.temperature if temperature is None else temperature),
        float(args.top_p if top_p is None else top_p),
        device,
    )
    return tokenizer.decode(seqs[0][0, plen:], skip_special_tokens=True)
