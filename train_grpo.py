#!/usr/bin/env python3
"""
On-policy GRPO-style training for the local Qwen filer and verifier (same backbone, two prompts).

Roll out K sampled completions per prompt, compute scalar rewards, group-normalize advantages,
then minimize  -adv * sum_t log pi(a_t) + beta * KL(pi || pi_ref)  over completion tokens.

With ``--alternate-rounds``, **both** filer and verifier are trained every step (same rollouts
as without the flag). Odd/even steps only **flip which agent's GRPO backward runs first**
(filer-first vs verifier-first), which can matter for optimizer interaction on a shared backbone.

Each training step is appended to ``<output-dir>/<run-name>/metrics.jsonl`` (one JSON object
per line) and ``rounds.tsv`` for quick viewing. Checkpoints are written under the same
``run-name`` folder when ``--use-lora`` is set.

Use ``--resume-from <previous-run-dir>`` to load ``adapter_final/`` and continue training.
If ``training_state.pt`` exists in that directory, the optimizer (and scaler) and global
step counter are restored so training picks up where it left off.

With ``--case-taxonomy --line-case-taxonomy``, each TaxCalcBench line gets a case id from
per-line x / y (SAC) / y' (full) and global revise flags; GRPO rewards are shifted to favor
cases 1 and 5 over 2–4 and 6.

Requires: torch, transformers, peft (recommended for KL via disable_adapter), lxml.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from device_pick import resolve_torch_device
from tax_case_taxonomy import (
    CaseTaxonomyConfig,
    build_revise_feedback_dict,
    classify_episode,
    default_case_loss_weights,
    drafts_near_identical,
    dual_verifier_rewards_from_raw,
)
from tax_line_case import (
    LineCaseConfig,
    case_preference_score,
    classify_line_case_id,
    line_case_loss_weights,
    per_line_case_counts_from_verifier_raw,
)

from agents import (
    _trim_draft_for_verifier,
    parse_verifier_raw_to_safety_case,
)
from client import extract_json_from_response
from dataset_registry import get_dataset_config, gold_path_for_case
import line_eval
from rewards import (
    line_verdict_binary_aligned,
    reward_filer_draft,
    reward_verifier_safety_case,
)
from schemas import DraftLine, DraftReturn
from line_eval import evaluated_line_ids, line_strict_correctness_by_line_id

_ACTIVE_DATASET = "tax"


def configure_dataset(name: str) -> None:
    global _ACTIVE_DATASET
    _ACTIVE_DATASET = name
    line_eval.set_active_dataset(name)


def _ds_cfg():
    return get_dataset_config(_ACTIVE_DATASET)


def _gold_path(case_dir: Path) -> Path:
    return gold_path_for_case(case_dir, _ACTIVE_DATASET)


def _filer_line_prompt() -> str:
    return _ds_cfg().filer_line_system_prompt


def _approver_line_prompt() -> str:
    return _ds_cfg().approver_line_system_prompt


def _verifier_system_prompt_for(args: argparse.Namespace) -> str:
    """Linewise mode → single-line verifier prompt; otherwise full-return prompt."""
    ds = _ds_cfg()
    if bool(getattr(args, "linewise_rollout", False)):
        return ds.verifier_line_system_prompt
    return ds.verifier_system_prompt

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False
    PeftModel = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


def _save_training_state(
    run_dir: Path,
    last_step: int,
    optimizer: AdamW,
    scaler: Optional[Any],
) -> None:
    payload: Dict[str, Any] = {
        "last_step": last_step,
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, run_dir / "training_state.pt")


def _assert_adapter_artifacts(save_dir: Path) -> None:
    has_weight = (save_dir / "adapter_model.safetensors").exists() or (
        save_dir / "adapter_model.bin"
    ).exists()
    if not has_weight:
        raise RuntimeError(
            f"Adapter weights were not saved in {save_dir} "
            "(expected adapter_model.safetensors or adapter_model.bin)"
        )
    if not (save_dir / "adapter_config.json").exists():
        raise RuntimeError(f"Missing adapter_config.json in {save_dir}")


def _save_lora_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    run_dir: Path,
    save_dir: Path,
    step_id: int,
    optimizer: AdamW,
    scaler: Optional[Any],
) -> None:
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    _assert_adapter_artifacts(save_dir)
    _save_training_state(run_dir, step_id, optimizer, scaler)
    if not (run_dir / "training_state.pt").exists():
        raise RuntimeError(f"Missing training_state.pt in {run_dir}")


def _resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if (p / "snapshots").exists():
        snaps = list((p / "snapshots").iterdir())
        if snaps:
            return str(snaps[0])
    return str(p)


def discover_case_dirs(root: Path, gold_filename: Optional[str] = None) -> List[Path]:
    """Directories under ``root`` that contain ``input.json`` and the gold artifact."""
    gold_name = gold_filename or _ds_cfg().gold_filename
    out: List[Path] = []
    for input_path in root.rglob("input.json"):
        d = input_path.parent
        if (d / gold_name).exists():
            out.append(d)
    return sorted(set(out))


def _select_case_for_step(
    cases: List[Path], step: int, cycle: bool, rng: random.Random
) -> Path:
    """Pick the training case for a given 1-indexed step.

    If ``cycle`` is True, iterate through ``cases`` deterministically in sorted
    order using ``(step - 1) % len(cases)`` so reruns are byte-identical. Else
    draw a random case from the pool (current default).
    """
    if not cases:
        raise ValueError("No cases provided to _select_case_for_step")
    if cycle:
        return cases[(max(int(step), 1) - 1) % len(cases)]
    return rng.choice(cases)


def split_train_test_cases(
    cases: List[Path], test_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if not cases:
        return [], []
    frac = max(0.0, min(0.95, float(test_fraction)))
    if frac <= 0.0 or len(cases) < 2:
        return list(cases), []
    n_test = int(round(len(cases) * frac))
    n_test = max(1, min(len(cases) - 1, n_test))
    ordered = sorted(cases, key=lambda p: str(p.resolve()))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    test_cases = sorted(ordered[:n_test], key=lambda p: str(p.resolve()))
    train_cases = sorted(ordered[n_test:], key=lambda p: str(p.resolve()))
    return train_cases, test_cases


def load_case_input(case_dir: Path) -> Dict[str, Any]:
    with (case_dir / "input.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)


def _draft_to_dict(d: Optional[DraftReturn]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {
        "return_version": d.return_version,
        "lines": [
            {
                "form": x.form,
                "line": x.line,
                "description": x.description,
                "amount": x.amount,
                "rationale": x.rationale,
            }
            for x in d.lines
        ],
        "metadata": d.metadata,
    }


def _write_step_trace(trace_dir: Path, gstep: int, payload: Dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    p = trace_dir / f"step_{gstep:05d}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _lora_target_modules(model: torch.nn.Module) -> List[str]:
    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found: set[str] = set()
    for n, _ in model.named_modules():
        for tail in names:
            if n.endswith(tail):
                found.add(tail)
    return sorted(found)


def build_chat_text(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    # Qwen3 exposes an ``enable_thinking`` switch on its chat template. When True
    # (the default) the template auto-inserts a <think> prelude, which the model
    # then fills with long chain-of-thought text that exhausts the token budget
    # and truncates the JSON output we actually want. We disable it here so the
    # model is expected to emit JSON directly.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def try_parse_draft(text: str) -> Optional[DraftReturn]:
    data = extract_json_from_response(text.strip())
    if not data or "lines" not in data:
        return None
    try:
        lines = [DraftLine(**line) for line in data.get("lines", [])]
    except (TypeError, ValueError):
        return None
    return DraftReturn(
        return_version=data.get("return_version", "ty24-v1"),
        lines=lines,
        metadata=data.get("metadata", {}),
    )


def try_parse_verifier_safety_case(text: str, draft: DraftReturn) -> Optional[Any]:
    data = extract_json_from_response(text.strip())
    if not data or "line_findings" not in data:
        return None
    try:
        return parse_verifier_raw_to_safety_case(data, draft, sac_only=False)
    except (TypeError, ValueError, KeyError):
        return None


def try_parse_line_draft(text: str, target_line: str) -> Optional[DraftLine]:
    """Parse one-line filer output JSON into a ``DraftLine``."""
    data = extract_json_from_response(text.strip())
    if not data:
        return None
    if isinstance(data.get("lines"), list):
        for item in data.get("lines", []):
            try:
                d = DraftLine(**item)
                if str(d.line).strip() == str(target_line):
                    return d
            except Exception:
                continue
    needed = {"form", "line", "description", "amount", "rationale"}
    if needed.issubset(set(data.keys())):
        try:
            d = DraftLine(
                form=str(data.get("form", "1040")),
                line=str(data.get("line", "")),
                description=str(data.get("description", "")),
                amount=float(data.get("amount", 0.0)),
                rationale=str(data.get("rationale", "")),
            )
            if str(d.line).strip() == str(target_line):
                return d
        except Exception:
            return None
    return None


def _line_filer_user_payload(
    input_json: Dict[str, Any], line_id: str, context_so_far: List[Dict[str, Any]]
) -> str:
    return _ds_cfg().line_filer_user_payload(input_json, line_id, context_so_far)


def _line_verifier_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_verifier_user_payload(
        input_json, line_obj, line_id, context_so_far
    )


def _line_revise_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_revise_user_payload(
        input_json, line_obj, line_id, verifier_raw, context_so_far
    )


def _line_approver_user_payload(
    input_json: Dict[str, Any],
    line_obj: DraftLine,
    line_id: str,
    verifier_raw: Dict[str, Any],
    revised_line: Optional[DraftLine],
    context_so_far: List[Dict[str, Any]],
) -> str:
    return _ds_cfg().line_approver_user_payload(
        input_json, line_obj, line_id, verifier_raw, revised_line, context_so_far
    )


def _parse_approver_decision(text: str) -> Optional[str]:
    """Return 'KEEP', 'REVISE', or None when the model output is unparseable."""
    raw = extract_json_from_response((text or "").strip()) or {}
    if isinstance(raw, dict):
        d = raw.get("decision")
        if isinstance(d, str):
            d_up = d.strip().upper()
            if d_up in ("KEEP", "REVISE"):
                return d_up
    return None


def _extract_line_verdict_from_raw(raw: Dict[str, Any], line_id: str) -> str:
    lfs = raw.get("line_findings", [])
    if isinstance(lfs, list):
        for lf in lfs:
            if str(lf.get("line", "")).strip() == str(line_id):
                return str(lf.get("verdict", "missing")).strip() or "missing"
    v = raw.get("verdict")
    if isinstance(v, str) and v:
        return v
    return "missing"


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


@torch.enable_grad()
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
) -> float:
    """One GRPO group: backward **per sequence** so peak memory is O(1) seq graph, not O(K).

    Gradients accumulate to match the mean of per-sequence GRPO losses. Returns that mean
    (detached) for logging.
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


@torch.enable_grad()
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


@torch.no_grad()
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


def _set_active_adapter(model: torch.nn.Module, adapter_name: str) -> None:
    if _HAS_PEFT and PeftModel is not None and isinstance(model, PeftModel):
        model.set_adapter(adapter_name)


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


@torch.no_grad()
def _run_dual_periodic_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    test_cases: List[Path],
    args: argparse.Namespace,
    device: str,
    gstep: int,
    eval_fp: Any,
) -> Optional[Dict[str, Any]]:
    """Run periodic eval on the held-out test split.

    Returns the metrics dict that was just written to ``eval_fp`` (or None
    when no test cases are configured). The caller uses the dict for
    best-checkpoint tracking and early-stopping decisions.
    """
    if not test_cases:
        return None
    max_cases = int(args.eval_max_cases)
    selected = test_cases[: max_cases if max_cases > 0 else len(test_cases)]
    if not selected:
        return
    prev_training = model.training
    prev_temp = float(args.temperature)
    prev_top_p = float(args.top_p)
    args.temperature = float(args.eval_temperature)
    args.top_p = float(args.eval_top_p)
    model.eval()
    recs: List[Dict[str, Any]] = []
    for case_dir in selected:
        input_json = load_case_input(case_dir)
        gold_path = _gold_path(case_dir)
        transitions, draft_x, draft_x_post_revise = _dual_linewise_rollout_transitions(
            model, tokenizer, input_json, gold_path, args, device, gstep
        )
        # 8-bucket linewise taxonomy {1, 2, 3, 4, 5a, 5b, 6a, 6b}, mutually
        # exclusive. See _dual_linewise_rollout_transitions for definitions.
        case_hist: Dict[str, int] = {
            "1": 0, "2": 0, "3": 0, "4": 0,
            "5a": 0, "5b": 0, "6a": 0, "6b": 0,
        }
        for tr in transitions:
            cid = int(tr.get("case_id", 0))
            csub = str(tr.get("case_sub", "other"))
            if cid in (1, 2, 3, 4):
                case_hist[str(cid)] += 1
            elif cid in (5, 6) and csub in ("5a", "5b", "6a", "6b"):
                case_hist[csub] += 1
        acc = float(reward_filer_draft(draft_x, gold_path))
        acc_post = float(reward_filer_draft(draft_x_post_revise, gold_path))
        lc = line_strict_correctness_by_line_id(draft_x, gold_path)
        lc_post = line_strict_correctness_by_line_id(draft_x_post_revise, gold_path)
        line_acc = (
            sum(1.0 for _lid, ok in lc.items() if ok) / max(len(lc), 1) if lc else 0.0
        )
        line_acc_post = (
            sum(1.0 for _lid, ok in lc_post.items() if ok) / max(len(lc_post), 1)
            if lc_post
            else 0.0
        )
        recs.append(
            {
                "case_dir": str(case_dir.resolve()),
                "final_draft_accuracy_strict": acc,
                "final_draft_accuracy_post_revise": acc_post,
                "final_draft_line_accuracy": float(line_acc),
                "final_draft_line_accuracy_post_revise": float(line_acc_post),
                "transitions_evaluated": len(transitions),
                "case_histogram": case_hist,
            }
        )
    agg_hist: Dict[str, int] = {
        "1": 0, "2": 0, "3": 0, "4": 0,
        "5a": 0, "5b": 0, "6a": 0, "6b": 0,
    }
    for r in recs:
        h = r["case_histogram"]
        for k in agg_hist.keys():
            agg_hist[k] += int(h.get(k, 0))
    out = {
        "step": int(gstep),
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_test_cases_evaluated": len(recs),
        "test_accuracy_strict_mean": float(
            sum(float(r["final_draft_accuracy_strict"]) for r in recs) / max(len(recs), 1)
        ),
        "test_accuracy_strict_post_revise_mean": float(
            sum(float(r["final_draft_accuracy_post_revise"]) for r in recs)
            / max(len(recs), 1)
        ),
        "test_line_accuracy_mean": float(
            sum(float(r["final_draft_line_accuracy"]) for r in recs) / max(len(recs), 1)
        ),
        "test_line_accuracy_post_revise_mean": float(
            sum(float(r["final_draft_line_accuracy_post_revise"]) for r in recs)
            / max(len(recs), 1)
        ),
        "test_case_histogram": agg_hist,
        "per_case": recs,
    }
    eval_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
    eval_fp.flush()
    args.temperature = prev_temp
    args.top_p = prev_top_p
    if prev_training:
        model.train()
    return out


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
            # Filer-only baseline path: sample K=2 paired drafts, reward each
            # by oracle line correctness S_x, build a paired-action GRPO
            # transition for the Filer adapter and skip Verifier+Approver.
            # The "best" draft (or first if tied) is used for context_so_far
            # to keep the per-line cascade consistent with the multi-agent
            # rollout's selection rule.
            pair_texts: List[str] = []
            pair_lines: List[Optional[DraftLine]] = []
            pair_S: List[float] = []
            for _k in range(2):
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
                pair_texts.append(cand_text)
                pair_lines.append(cand_line)
                pair_S.append(cand_S)
            best_idx = 0 if pair_S[0] >= pair_S[1] else 1
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
            # Best-of-K revise sampling. Draw K candidates at an elevated
            # temperature (--revise-temperature) and select by:
            #   1) S_z == 1 (lands gold) ─ populates 5a / 6b buckets
            #   2) parse-valid           ─ avoids garbage revise text
            #   3) first sample          ─ fallback
            # This both (a) gives the Approver a higher-quality revise to
            # judge and (b) gives the filer's revise role more useful gradient.
            revise_k = max(int(getattr(args, "revise_sample_k", 1) or 1), 1)
            revise_temp = float(getattr(args, "revise_temperature", args.temperature))
            best_text = ""
            best_line: Optional[DraftLine] = None
            best_score = -1  # 2 = lands gold, 1 = parses, 0 = garbage
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
                    score = 0
                else:
                    cand_tmp = DraftReturn(
                        return_version=_ds_cfg().return_version, lines=[cand_line], metadata={}
                    )
                    lands = bool(
                        line_strict_correctness_by_line_id(cand_tmp, gold_path).get(
                            lid, False
                        )
                    )
                    score = 2 if lands else 1
                if score > best_score:
                    best_score = score
                    best_text = cand_text
                    best_line = cand_line
                    if score == 2:
                        break  # can't do better than landing gold
            revise_text = best_text
            z_line = best_line
            revise_parsed = z_line is not None
            if z_line is not None:
                S_z = 1.0 if best_score == 2 else 0.0

            # Counterfactual REVISE-generation reward (used to train the filer to
            # produce better revise text, conditional on KEEP vs REVISE prompts).
            if sac_correct_line:
                cf_keep, cf_revise = 0.0, 1.0
            else:
                cf_keep, cf_revise = 1.0, 0.0
            keep_reward = cf_keep + outcome_bonus * float(S_x)
            if z_line is not None:
                revise_reward = cf_revise + outcome_bonus * float(S_z)
            else:
                # Invalid revise JSON: keep counterfactual decision reward but apply
                # a small format penalty so producing garbage isn't free.
                revise_reward = cf_revise - fmt_pen

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
            if ap_eps > 0.0 and revision_decision != "REVISE":
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
            if len(samples) >= 2 and len(rewards) >= 2 and f_user:
                _set_active_adapter(model, roles["filer_adapter"])
                seqs_f: List[torch.Tensor] = []
                pl_f: List[int] = []
                for sample_text in samples[:2]:
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
                    rewards[:2], device=device, dtype=torch.float32
                )
                # Skip update when both samples have identical reward: the
                # GRPO advantage is zero everywhere and the gradient is zero
                # too (advantage standardisation is undefined for std=0).
                # This also serves as DAPO's dynamic-sampling filter (skip
                # groups whose rollouts share the same reward — no learning
                # signal).
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

        # Filer KEEP/REVISE paired update. Skip when the revise output was invalid
        # JSON: training on that noisy text makes KEEP structurally dominate and
        # fights any real revision signal.
        do_filer_revise_update = bool(tr["revision"]["considered"]) and not (
            skip_f_on_invalid_revise and revise_parsed is False
        )
        if do_filer_revise_update:
            _set_active_adapter(model, roles["filer_adapter"])
            f_user = tr["prompts"]["revise_user"]
            keep_text = tr["revision"]["keep_text"]
            revise_text = tr["revision"]["revise_text"] or keep_text
            seq_keep, pl_keep = _build_seq_from_prompt_and_action(
                tokenizer, _filer_line_prompt(), f_user, keep_text, device
            )
            seq_rev, pl_rev = _build_seq_from_prompt_and_action(
                tokenizer, _filer_line_prompt(), f_user, revise_text, device
            )
            rfs = torch.tensor(
                [tr["revision"]["keep_reward"], tr["revision"]["revise_reward"]],
                device=device,
                dtype=torch.float32,
            )
            adv_f = (rfs - rfs.mean()) / (rfs.std(unbiased=False) + 1e-8)
            lf_total += grpo_group_accumulate_gradients(
                model,
                [seq_keep, seq_rev],
                [pl_keep, pl_rev],
                adv_f,
                args.beta_kl,
                device,
                use_amp,
                scaler,
                loss_scale=1.0,
            )
            n_backward += 1

        # Approver KEEP/REVISE decision-token paired update. Trains the
        # *Approver adapter* (pinned, see _role_assignment) to emit the right
        # meta-action under _approver_line_prompt() when the verifier has
        # flagged a line. Always trained (counterfactual) whenever a revision
        # was considered: even when the revise sample was garbage, the right
        # answer ("KEEP, the revision is unusable") is informative.
        ap = tr.get("approver", {}) or {}
        if bool(ap.get("considered", False)):
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


def train_loop(args: argparse.Namespace) -> None:
    configure_dataset(getattr(args, "dataset", "tax"))
    device = resolve_torch_device()
    print(f"[train_grpo] Device: {device}")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model_path = _resolve_model_path(args.model_path)
    print(f"[train_grpo] Loading tokenizer/model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.train()

    resume_from = (getattr(args, "resume_from", "") or "").strip()
    adapter_resume: Optional[Path] = None
    if resume_from:
        if not args.use_lora:
            raise SystemExit("--resume-from requires --use-lora")
        adapter_resume = Path(resume_from).resolve() / "adapter_final"
        if not adapter_resume.is_dir():
            raise SystemExit(
                f"--resume-from: expected LoRA checkpoint at {adapter_resume} (save a run with --use-lora first)"
            )

    if args.use_lora:
        if not _HAS_PEFT:
            raise RuntimeError("peft is required for --use-lora. pip install peft")
        if adapter_resume is not None:
            print(f"[train_grpo] Resuming LoRA from {adapter_resume}")
            model = PeftModel.from_pretrained(
                model, str(adapter_resume), is_trainable=True
            )
            model.print_trainable_parameters()
        else:
            targets = _lora_target_modules(model)
            if not targets:
                raise RuntimeError("Could not infer LoRA target modules for this model.")
            print(f"[train_grpo] LoRA targets: {targets}")
            lcfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            model = get_peft_model(model, lcfg)
            if getattr(args, "dual_agent_replay", False):
                try:
                    model.add_adapter("theta", lcfg)
                    model.set_adapter("default")
                    print("[train_grpo] Dual adapters enabled: phi=default, theta=theta")
                except Exception as e:
                    raise RuntimeError(
                        f"Could not create second adapter 'theta' for dual-agent mode: {e}"
                    )
            model.print_trainable_parameters()
    else:
        if resume_from:
            raise SystemExit("--resume-from requires --use-lora")
        print("[train_grpo] Warning: training full weights without LoRA is heavy; consider --use-lora")

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[train_grpo] Gradient checkpointing on (lower VRAM, slower steps)")
        except Exception as e:
            print(f"[train_grpo] Warning: could not enable gradient checkpointing: {e}")

    # Collect *all* LoRA parameters across every adapter into the optimizer.
    # PEFT's set_adapter() toggles requires_grad on adapter LoRA layers (active
    # -> True, others -> False), so a naive `[p for p in model.parameters() if
    # p.requires_grad]` only captures the currently-active adapter. In dual-
    # agent mode we switch the active adapter mid-step (verifier <-> filer/
    # approver), and gradients from the inactive-at-init adapter would land on
    # parameters the optimizer never saw -- training nothing AND triggering
    # AMP's "no inf checks were recorded" assertion when an entire sub-batch
    # only ever activated the not-in-optimizer adapter (e.g. a case with zero
    # SAC raises only runs the verifier path).
    trainable = []
    seen_ids = set()
    if getattr(args, "use_lora", False) and getattr(args, "dual_agent_replay", False):
        prev_active = getattr(model, "active_adapter", "default")
        for adapter_name in ("default", "theta"):
            try:
                model.set_adapter(adapter_name)
            except Exception:
                continue
            for p in model.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    trainable.append(p)
                    seen_ids.add(id(p))
        # Restore active adapter so downstream code starts from a known state.
        try:
            model.set_adapter(prev_active)
        except Exception:
            model.set_adapter("default")
        # Ensure every adapter LoRA param has requires_grad=True at optimizer-
        # creation time. PEFT will still toggle this per active adapter during
        # training, but inactive adapters whose grads remain None are simply
        # skipped by optimizer.step (and don't trigger AMP assertions, since
        # at least one active-adapter param will have a grad in any non-empty
        # backward pass).
        for p in trainable:
            p.requires_grad = True
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[train_grpo] Optimizer trainable param tensors: {len(trainable)} "
        f"({sum(p.numel() for p in trainable):,} elements)"
    )
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler: Optional[Any] = None
    # GradScaler is needed for fp16 (to prevent underflow) but is incompatible
    # with bf16 — the unscale kernel `_amp_foreach_non_finite_check_and_unscale_cuda`
    # is fp16-only. We always use bf16 on CUDA (set above), and bf16 has the same
    # dynamic range as fp32 so loss scaling is unnecessary anyway.
    if args.amp and device.startswith("cuda") and dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()
    if scaler is None and args.amp and device.startswith("cuda"):
        print(
            "[train_grpo] AMP autocast on (bf16); GradScaler skipped "
            "(bf16 doesn't need loss scaling and the unscale kernel is fp16-only)."
        )

    global_step_base = 0
    if resume_from:
        state_pt = Path(resume_from).resolve() / "training_state.pt"
        if state_pt.is_file():
            sd = torch.load(state_pt, map_location=device)
            try:
                optimizer.load_state_dict(sd["optimizer"])
            except (ValueError, RuntimeError) as e:
                # Pre-fix checkpoints had only one adapter's params in the
                # optimizer; the new optimizer has both adapters. Skip loading
                # rather than crashing -- training resumes with fresh AdamW
                # state and the LoRA weights themselves still come from the
                # adapter directory.
                print(
                    f"[train_grpo] Could not load optimizer state from {state_pt} "
                    f"(likely a param-count mismatch from the dual-adapter fix): {e}\n"
                    f"[train_grpo] Continuing with fresh optimizer state."
                )
            if scaler is not None and sd.get("scaler") is not None:
                try:
                    scaler.load_state_dict(sd["scaler"])
                except Exception as e:
                    print(f"[train_grpo] Could not load scaler state: {e}")
            global_step_base = int(sd.get("last_step", 0))
            print(
                f"[train_grpo] Loaded training state from {state_pt} (last_step={global_step_base})"
            )
        else:
            print(
                f"[train_grpo] No {state_pt.name} found — fresh optimizer; only LoRA weights were restored"
            )

    cases_all = discover_case_dirs(Path(args.cases_root))
    if not cases_all:
        raise SystemExit(
            f"No cases with input.json+{_ds_cfg().gold_filename} under {args.cases_root}"
        )
    train_cases, test_cases = split_train_test_cases(
        cases_all, args.test_fraction, args.split_seed
    )
    if not train_cases:
        raise SystemExit("Train split is empty. Reduce --test-fraction.")
    print(f"[train_grpo] Found {len(cases_all)} cases under {args.cases_root}")
    print(
        f"[train_grpo] Split: train={len(train_cases)} test={len(test_cases)} "
        f"(test_fraction={args.test_fraction}, split_seed={args.split_seed})"
    )
    full_train_cases = list(train_cases)
    subset_size = int(getattr(args, "train_case_subset_size", 0) or 0)
    resample_each_step = bool(getattr(args, "train_case_resample_each_step", False))
    if subset_size > 0:
        subset_size = min(subset_size, len(train_cases))
        if resample_each_step:
            # Keep `train_cases` = the FULL pool. Per-step subsetting happens in
            # the training loop. This avoids the memorisation pathology (cycling
            # 3-5 of 41 train cases caused tenth_run to plateau at step ~2/30).
            train_cases = list(full_train_cases)
            print(
                f"[train_grpo] Train case subset: per-step random {subset_size} / "
                f"{len(full_train_cases)} cases (resample-each-step=ON; pool size "
                f"= {len(full_train_cases)})"
            )
        else:
            train_cases = list(train_cases)[:subset_size]
            print(
                f"[train_grpo] Train case subset: using {len(train_cases)} / "
                f"{len(full_train_cases)} cases "
                f"(cycle={bool(args.train_case_cycle)})"
            )
            for p in train_cases:
                print(f"[train_grpo]   - {p.name}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (args.run_name or "").strip() or datetime.now(timezone.utc).strftime(
        "run_%Y%m%d_%H%M%S"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "step_traces"

    run_meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "cases_root": str(Path(args.cases_root).resolve()),
        "n_cases": len(cases_all),
        "n_train_cases_full": len(full_train_cases),
        "n_train_cases": len(train_cases),
        "n_test_cases": len(test_cases),
        "train_case_dirs_full": [str(p.resolve()) for p in full_train_cases],
        "train_case_dirs": [str(p.resolve()) for p in train_cases],
        "test_case_dirs": [str(p.resolve()) for p in test_cases],
        "train_case_subset_size": subset_size,
        "train_case_cycle": bool(args.train_case_cycle),
        "run_dir": str(run_dir.resolve()),
        "resume_from": resume_from or None,
        "global_step_base": global_step_base,
        "args": vars(args),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    tsv_path = run_dir / "rounds.tsv"
    print(f"[train_grpo] Per-step logs -> {metrics_path}")
    print(f"[train_grpo] Eval logs       -> {eval_metrics_path}")
    print(f"[train_grpo] TSV summary     -> {tsv_path}")

    rng = random.Random(args.seed)
    use_amp = scaler is not None and device.startswith("cuda")

    metrics_fp = open(metrics_path, "w", encoding="utf-8")
    eval_fp = open(eval_metrics_path, "w", encoding="utf-8")
    tsv_fp = open(tsv_path, "w", encoding="utf-8")
    tsv_fp.write(
        "global_step\tround\tcase_dir\tloss\tfiler_r_mean\tver_r_mean\tskipped\tnotes\n"
    )

    try:
        _training_steps(
            args,
            model,
            tokenizer,
            optimizer,
            scaler,
            train_cases,
            rng,
            device,
            use_amp,
            run_dir,
            trace_dir,
            metrics_fp,
            tsv_fp,
            eval_fp,
            global_step_base,
            test_cases,
        )
    finally:
        metrics_fp.close()
        eval_fp.close()
        tsv_fp.close()
        print(f"[train_grpo] Run artifacts in {run_dir.resolve()}")


def _training_steps(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: AdamW,
    scaler: Optional[Any],
    cases: List[Path],
    rng: random.Random,
    device: str,
    use_amp: bool,
    run_dir: Path,
    trace_dir: Path,
    metrics_fp: Any,
    tsv_fp: Any,
    eval_fp: Any,
    global_step_base: int,
    test_cases: List[Path],
) -> None:
    if getattr(args, "dual_agent_replay", False):
        _training_steps_dual(
            args,
            model,
            tokenizer,
            optimizer,
            scaler,
            cases,
            rng,
            device,
            use_amp,
            run_dir,
            trace_dir,
            metrics_fp,
            tsv_fp,
            eval_fp,
            global_step_base,
            test_cases,
        )
        return
    for step in range(1, args.steps + 1):
        gstep = global_step_base + step
        case_dir = _select_case_for_step(
            cases, step, bool(getattr(args, "train_case_cycle", False)), rng
        )
        input_json = load_case_input(case_dir)
        gold_path = _gold_path(case_dir)

        # Same rollouts every step: filer K samples → best draft → verifier K samples.
        lf = 0.0
        lv = 0.0
        r_f_mean = 0.0
        r_v_mean = 0.0
        r_f_list: Optional[List[float]] = None
        r_v_list: Optional[List[float]] = None
        notes = ""
        seqs_f: Optional[List[torch.Tensor]] = None
        prompt_len_f = 0
        prompt_lens_f: Optional[List[int]] = None
        adv_f: Optional[torch.Tensor] = None
        seqs_v: Optional[List[torch.Tensor]] = None
        prompt_len_v = 0
        prompt_lens_v: Optional[List[int]] = None
        adv_v: Optional[torch.Tensor] = None
        texts_f: List[str] = []
        texts_v: List[str] = []
        case_id = -1
        wf, wv = 1.0, 1.0
        z_acc_log: Optional[float] = None
        revise_keep_log: Optional[bool] = None
        line_case_counts: Optional[Dict[str, int]] = None
        line_case_pref: Optional[float] = None
        z_text_log: Optional[str] = None
        linewise_details: Optional[List[Dict[str, Any]]] = None

        if getattr(args, "linewise_rollout", False):
            (
                seqs_f,
                plf,
                adv_f,
                r_f,
                seqs_v,
                plv,
                adv_v,
                r_v,
                draft_x,
                lw_counts,
                lw_details,
            ) = linewise_filer_verifier_rollouts(
                model, tokenizer, input_json, gold_path, args, device
            )
            linewise_details = lw_details
            prompt_lens_f = plf
            prompt_lens_v = plv
            r_f_mean = float(r_f.mean().item())
            r_v_mean = float(r_v.mean().item())
            r_f_list = [float(x) for x in r_f.detach().cpu().tolist()]
            r_v_list = [float(x) for x in r_v.detach().cpu().tolist()]
            drafts = [draft_x]
            best_idx = 0
            if getattr(args, "line_case_taxonomy", False):
                pref = case_preference_score(lw_counts)
                line_case_pref = pref
                line_case_counts = {str(k): int(v) for k, v in lw_counts.items()}
                boost = float(args.line_case_reward_scale) * pref
                r_f = r_f + boost
                r_v = r_v + boost
                adv_f = (r_f - r_f.mean()) / (r_f.std(unbiased=False) + 1e-8)
                adv_v = (r_v - r_v.mean()) / (r_v.std(unbiased=False) + 1e-8)
                wf, wv = line_case_loss_weights(
                    lw_counts,
                    base_wf=1.0,
                    base_wv=1.0,
                    prefer_boost=float(args.line_case_prefer_boost),
                )
                case_id = -1
        else:
            seqs_f, prompt_len_f, adv_f, r_f, drafts, texts_f = filer_grpo_rollouts(
                model, tokenizer, input_json, gold_path, args, device
            )
            prompt_lens_f = [prompt_len_f] * len(seqs_f)
            r_f_mean = float(r_f.mean().item())
            r_f_list = [float(x) for x in r_f.detach().cpu().tolist()]
            best_idx = int(torch.argmax(r_f).item())
            draft_x = drafts[best_idx]
            if draft_x is None:
                draft_x = next((d for d in drafts if d is not None), None)
        if draft_x is not None:
            if not getattr(args, "linewise_rollout", False):
                seqs_v, prompt_len_v, adv_v, r_v, texts_v = verifier_grpo_rollouts(
                    model,
                    tokenizer,
                    input_json,
                    draft_x,
                    gold_path,
                    args,
                    device,
                )
                prompt_lens_v = [prompt_len_v] * len(seqs_v)
                r_v_mean = float(r_v.mean().item())
                r_v_list = [float(x) for x in r_v.detach().cpu().tolist()]
            if getattr(args, "case_taxonomy", False) and texts_v:
                x_acc = reward_filer_draft(draft_x, gold_path)
                bi = int(torch.argmax(r_v).item())
                raw_v = extract_json_from_response(texts_v[bi].strip())
                if raw_v:
                    r_sac, r_full = dual_verifier_rewards_from_raw(
                        raw_v, draft_x, gold_path
                    )
                    tc = CaseTaxonomyConfig(
                        x_good=args.case_x_good,
                        verifier_good=args.case_verifier_good,
                        y_prime_gt_y_margin=args.case_yprime_margin,
                        revise_draft_amount_abs_tol=args.case_revise_draft_amount_abs_tol,
                    )
                    z_acc_t: Optional[float] = None
                    rchx: Optional[bool] = None
                    z_draft: Optional[DraftReturn] = None
                    if not getattr(args, "no_case_taxonomy_revise", False):
                        z_acc_t, rchx, z_draft, z_text_log = revise_filer_sample_for_taxonomy(
                            model,
                            tokenizer,
                            input_json,
                            draft_x,
                            raw_v,
                            gold_path,
                            args,
                            device,
                            tc,
                        )
                        z_acc_log = z_acc_t
                        revise_keep_log = rchx
                    if getattr(args, "line_case_taxonomy", False):
                        lc = LineCaseConfig(
                            z_acc_improve_eps=args.line_case_z_improve_eps,
                            line_amount_tol=args.line_case_amount_tol,
                        )
                        counts, _line_rows, _ = per_line_case_counts_from_verifier_raw(
                            draft_x,
                            raw_v,
                            gold_path,
                            z_draft,
                            float(z_acc_t or 0.0),
                            revise_abs_tol=tc.revise_draft_amount_abs_tol,
                            line_cfg=lc,
                        )
                        pref = case_preference_score(counts)
                        line_case_pref = pref
                        line_case_counts = {str(k): int(v) for k, v in counts.items()}
                        boost = float(args.line_case_reward_scale) * pref
                        r_f = r_f + boost
                        r_v = r_v + boost
                        adv_f = (r_f - r_f.mean()) / (r_f.std(unbiased=False) + 1e-8)
                        adv_v = (r_v - r_v.mean()) / (r_v.std(unbiased=False) + 1e-8)
                        wf, wv = line_case_loss_weights(
                            counts,
                            base_wf=1.0,
                            base_wv=1.0,
                            prefer_boost=float(args.line_case_prefer_boost),
                        )
                        case_id = -1
                    else:
                        case_id = classify_episode(
                            x_acc,
                            r_sac,
                            r_full,
                            z_acc=z_acc_t,
                            revise_chose_keep_x=rchx,
                            cfg=tc,
                        )
                        wf, wv = default_case_loss_weights(case_id)
                else:
                    case_id = 0
                    wf, wv = default_case_loss_weights(0)
        else:
            notes = "no_valid_draft_for_verifier"

        if args.alternate_rounds:
            parity = (step - 1) % 2
            if args.round_start == "verifier":
                verifier_grad_first = parity == 0
            else:
                verifier_grad_first = parity == 1
            round_tag = (
                "both_verifier_grad_first"
                if verifier_grad_first
                else "both_filer_grad_first"
            )
        else:
            verifier_grad_first = False
            round_tag = "both"

        optimizer.zero_grad(set_to_none=True)
        if args.alternate_rounds and verifier_grad_first:
            if seqs_v is not None and adv_v is not None:
                lv = grpo_group_accumulate_gradients(
                    model,
                    seqs_v,
                    (prompt_lens_v or ([prompt_len_v] * len(seqs_v))),
                    adv_v,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=wv,
                )
            if seqs_f is not None and adv_f is not None:
                lf = grpo_group_accumulate_gradients(
                    model,
                    seqs_f,
                    (prompt_lens_f or ([prompt_len_f] * len(seqs_f))),
                    adv_f,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=wf,
                )
        else:
            if seqs_f is not None and adv_f is not None:
                lf = grpo_group_accumulate_gradients(
                    model,
                    seqs_f,
                    (prompt_lens_f or ([prompt_len_f] * len(seqs_f))),
                    adv_f,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=wf,
                )
            if seqs_v is not None and adv_v is not None:
                lv = grpo_group_accumulate_gradients(
                    model,
                    seqs_v,
                    (prompt_lens_v or ([prompt_len_v] * len(seqs_v))),
                    adv_v,
                    args.beta_kl,
                    device,
                    use_amp,
                    scaler,
                    loss_scale=wv,
                )

        did_step = (seqs_f is not None and adv_f is not None) or (
            seqs_v is not None and adv_v is not None
        )
        if did_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            loss_scalar = lf + lv
        else:
            loss_scalar = 0.0

        skipped = not did_step

        log_record = {
            "step": gstep,
            "local_step": step,
            "utc": datetime.now(timezone.utc).isoformat(),
            "round": round_tag,
            "case_dir": str(case_dir.resolve()),
            "loss_total": loss_scalar,
            "loss_filer_term": lf,
            "loss_verifier_term": lv,
            "filer_reward_mean": r_f_mean,
            "verifier_reward_mean": r_v_mean,
            # Lists (never null in JSON): empty [] if that role had no rollouts this step.
            "filer_rewards": r_f_list if r_f_list is not None else [],
            "verifier_rewards": r_v_list if r_v_list is not None else [],
            "skipped_optimizer": skipped,
            "notes": notes or None,
            "gradient_accumulation_order": (
                "verifier_first"
                if args.alternate_rounds and verifier_grad_first
                else "filer_first"
            ),
            "case_taxonomy_id": case_id,
            "case_taxonomy_filer_w": wf,
            "case_taxonomy_verifier_w": wv,
            "case_taxonomy_revise_z_acc": z_acc_log,
            "case_taxonomy_revise_chose_keep_x": revise_keep_log,
            "line_case_counts": line_case_counts,
            "line_case_preference": line_case_pref,
        }
        metrics_fp.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        metrics_fp.flush()
        try:
            final_acc = None
            final_line_correct: Optional[Dict[str, bool]] = None
            if draft_x is not None:
                final_acc = float(reward_filer_draft(draft_x, gold_path))
                final_line_correct = line_strict_correctness_by_line_id(draft_x, gold_path)
            chosen_idx = (
                best_idx
                if best_idx < len(drafts) and drafts[best_idx] is not None
                else next((i for i, d in enumerate(drafts) if d is not None), -1)
            )
            step_trace = {
                "step": gstep,
                "utc": log_record["utc"],
                "case_dir": str(case_dir.resolve()),
                "notes": notes or None,
                "filer": {
                    "candidate_texts": texts_f,
                    "candidate_rewards": r_f_list or [],
                    "candidate_json": [_draft_to_dict(d) for d in drafts],
                    "selected_idx": chosen_idx,
                    "selected_json": _draft_to_dict(draft_x),
                },
                "verifier": {
                    "candidate_texts": texts_v,
                    "candidate_rewards": r_v_list or [],
                    "candidate_json": [extract_json_from_response(t.strip()) for t in texts_v],
                },
                "revision": {
                    "text": z_text_log,
                    "json": extract_json_from_response(z_text_log.strip())
                    if z_text_log
                    else None,
                    "z_acc": z_acc_log,
                    "keep_x": revise_keep_log,
                },
                "taxonomy": {
                    "case_id": case_id,
                    "line_case_counts": line_case_counts,
                    "line_case_preference": line_case_pref,
                },
                "final_draft_accuracy_strict": final_acc,
                "final_draft_line_correctness": final_line_correct,
                "linewise_details": linewise_details,
            }
            _write_step_trace(trace_dir, gstep, step_trace)
        except Exception as e:
            print(f"[train_grpo] Warning: failed to write step trace {gstep}: {e}")
        tsv_fp.write(
            f"{gstep}\t{round_tag}\t{case_dir.name}\t{loss_scalar:.6f}\t"
            f"{r_f_mean:.6f}\t{r_v_mean:.6f}\t{int(skipped)}\t{notes}\n"
        )
        tsv_fp.flush()

        if step % args.log_every == 0:
            print(
                f"[step global={gstep} local={step}/{args.steps}] round={round_tag} case={case_dir.name} "
                f"filer_r_mean={r_f_mean:.4f} ver_r_mean={r_v_mean:.4f} "
                f"loss={loss_scalar:.4f} (filer_term={lf:.4f} verifier_term={lv:.4f})"
            )

        if step % args.save_every == 0 and args.use_lora:
            save_path = run_dir / f"adapter_step_{gstep}"
            _save_lora_checkpoint(
                model, tokenizer, run_dir, save_path, gstep, optimizer, scaler
            )
            print(f"[train_grpo] Saved adapter + training_state to {save_path}")

    if args.use_lora:
        final = run_dir / "adapter_final"
        g_final = global_step_base + args.steps
        _save_lora_checkpoint(
            model, tokenizer, run_dir, final, g_final, optimizer, scaler
        )
        print(f"[train_grpo] Saved final adapter + training_state (step {g_final}) to {final}")


def _training_steps_dual(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: AdamW,
    scaler: Optional[Any],
    cases: List[Path],
    rng: random.Random,
    device: str,
    use_amp: bool,
    run_dir: Path,
    trace_dir: Path,
    metrics_fp: Any,
    tsv_fp: Any,
    eval_fp: Any,
    global_step_base: int,
    test_cases: List[Path],
) -> None:
    buffer: List[Dict[str, Any]] = []
    buf_stats_path = run_dir / "buffer_stats.jsonl"
    buf_stats_fp = open(buf_stats_path, "w", encoding="utf-8")
    train_all = bool(getattr(args, "train_all_cases_per_step", False))
    replay_disabled = bool(getattr(args, "disable_replay", False))
    if replay_disabled:
        print(
            "[train_grpo] Replay buffer DISABLED via --disable-replay: "
            "on-policy paired-action GRPO only."
        )
    resample_each_step = bool(getattr(args, "train_case_resample_each_step", False))
    subset_size = int(getattr(args, "train_case_subset_size", 0) or 0)
    try:
        for step in range(1, args.steps + 1):
            gstep = global_step_base + step

            # Build the list of cases this step will iterate through.
            if resample_each_step and subset_size > 0:
                # Fresh random subset from the full pool every step. `cases` here
                # is already the full train pool (see train_loop). Use rng.sample
                # so the same case never appears twice within a single step.
                k = min(subset_size, len(cases))
                step_cases = rng.sample(list(cases), k)
            elif train_all:
                step_cases = list(cases)
            else:
                step_cases = [
                    _select_case_for_step(
                        cases,
                        step,
                        bool(getattr(args, "train_case_cycle", False)),
                        rng,
                    )
                ]

            transitions: List[Dict[str, Any]] = []
            lf = 0.0
            lv = 0.0
            per_case_records: List[Dict[str, Any]] = []
            primary_case_dir: Path = step_cases[-1]

            for sub_idx, case_dir in enumerate(step_cases):
                input_json = load_case_input(case_dir)
                gold_path = _gold_path(case_dir)
                sub_transitions, draft_x, draft_x_post_revise = (
                    _dual_linewise_rollout_transitions(
                        model, tokenizer, input_json, gold_path, args, device, gstep
                    )
                )
                if not replay_disabled:
                    buffer.extend(sub_transitions)
                    if len(buffer) > int(args.replay_capacity):
                        del buffer[: len(buffer) - int(args.replay_capacity)]

                slf, slv = _apply_transition_batch_update(
                    model,
                    tokenizer,
                    optimizer,
                    scaler,
                    sub_transitions,
                    args,
                    device,
                    use_amp,
                )
                sub_acc = float(reward_filer_draft(draft_x, gold_path))
                sub_acc_post = float(reward_filer_draft(draft_x_post_revise, gold_path))
                sub_line_correct = line_strict_correctness_by_line_id(draft_x, gold_path)
                sub_line_correct_post = line_strict_correctness_by_line_id(
                    draft_x_post_revise, gold_path
                )
                per_case_records.append(
                    {
                        "sub_idx": sub_idx,
                        "case_dir": str(case_dir.resolve()),
                        "n_transitions": len(sub_transitions),
                        "loss_filer_term": float(slf),
                        "loss_verifier_term": float(slv),
                        "final_draft_accuracy_strict": sub_acc,
                        "final_draft_accuracy_post_revise": sub_acc_post,
                        "final_draft_line_correctness": sub_line_correct,
                        "final_draft_line_correctness_post_revise": sub_line_correct_post,
                    }
                )
                transitions.extend(sub_transitions)
                lf += slf
                lv += slv
                # Track the last case for legacy single-case log fields.
                primary_case_dir = case_dir

            # Replay happens once per step, using the combined buffer. Skipped
            # entirely when --disable-replay is set.
            replay_used = 0
            replay_batch_size = 0
            if (
                not replay_disabled
                and len(buffer) >= int(args.replay_warmup)
            ):
                recent = buffer[-int(args.replay_recent_window) :]
                for _ in range(int(args.replay_updates_per_iter)):
                    if not recent:
                        break
                    rb = min(int(args.replay_batch), len(recent))
                    replay = rng.sample(recent, rb) if len(recent) > rb else list(recent)
                    rlf, rlv = _apply_transition_batch_update(
                        model, tokenizer, optimizer, scaler, replay, args, device, use_amp
                    )
                    lf += rlf
                    lv += rlv
                    replay_used += 1
                    replay_batch_size = rb

            # 8-bucket linewise taxonomy {1, 2, 3, 4, 5a, 5b, 6a, 6b},
            # mutually exclusive — see _dual_linewise_rollout_transitions.
            case_hist: Dict[str, int] = {
                "1": 0, "2": 0, "3": 0, "4": 0,
                "5a": 0, "5b": 0, "6a": 0, "6b": 0,
            }
            filer_rewards_step: List[float] = []
            verifier_rewards_step: List[float] = []
            approver_rewards_step: List[float] = []
            n_lines = 0
            n_filer_invalid = 0
            n_sac = 0
            n_revise_considered = 0
            n_revise_invalid = 0
            n_revise_chosen = 0
            n_approver_invalid = 0
            n_decision_right = 0
            n_decision_wrong = 0
            n_forced_explore = 0
            for tr in transitions:
                cid = int(tr.get("case_id", 0))
                csub = str(tr.get("case_sub", "other"))
                if cid in (1, 2, 3, 4):
                    case_hist[str(cid)] += 1
                elif cid in (5, 6) and csub in ("5a", "5b", "6a", "6b"):
                    case_hist[csub] += 1
                y_choice = str(tr.get("y_choice", "NO_SAC"))
                vr = tr.get("verifier_rewards", {}) or {}
                verifier_rewards_step.append(float(vr.get(y_choice, 0.0)))
                rev = tr.get("revision", {}) or {}
                ap = tr.get("approver", {}) or {}
                if bool(rev.get("considered", False)) and str(rev.get("decision", "KEEP")) == "REVISE":
                    filer_rewards_step.append(float(rev.get("revise_reward", 0.0)))
                else:
                    filer_rewards_step.append(float(rev.get("keep_reward", 0.0)))
                if bool(ap.get("considered", False)):
                    if str(ap.get("decision", "KEEP")) == "REVISE":
                        approver_rewards_step.append(float(ap.get("revise_reward", 0.0)))
                    else:
                        approver_rewards_step.append(float(ap.get("keep_reward", 0.0)))
                    if ap.get("parsed") is False:
                        n_approver_invalid += 1
                    dwr = ap.get("decision_was_right")
                    if dwr is True:
                        n_decision_right += 1
                    elif dwr is False:
                        n_decision_wrong += 1
                    if bool(ap.get("forced_explore", False)):
                        n_forced_explore += 1
                n_lines += 1
                if not bool(tr.get("filer_parsed", True)):
                    n_filer_invalid += 1
                if y_choice == "SAC":
                    n_sac += 1
                if bool(rev.get("considered", False)):
                    n_revise_considered += 1
                    if rev.get("parsed") is False:
                        n_revise_invalid += 1
                    if str(rev.get("decision", "KEEP")) == "REVISE":
                        n_revise_chosen += 1
            denom = float(max(n_lines, 1))
            rev_denom = float(max(n_revise_considered, 1))
            sac_rate = float(n_sac) / denom
            invalid_filer_json_rate = float(n_filer_invalid) / denom
            invalid_revise_json_rate = float(n_revise_invalid) / rev_denom
            revise_rate = float(n_revise_chosen) / rev_denom
            invalid_approver_json_rate = float(n_approver_invalid) / rev_denom
            dec_total = float(max(n_decision_right + n_decision_wrong, 1))
            decision_accuracy = float(n_decision_right) / dec_total
            approver_forced_revise_rate = float(n_forced_explore) / rev_denom
            roles = _role_assignment(
                gstep,
                args.role_swap_period,
                swap_enabled=bool(getattr(args, "swap_roles", False)),
            )
            final_accs = [
                float(pc["final_draft_accuracy_strict"]) for pc in per_case_records
            ]
            final_acc = float(sum(final_accs) / max(len(final_accs), 1))
            final_accs_post = [
                float(pc["final_draft_accuracy_post_revise"])
                for pc in per_case_records
            ]
            final_acc_post = float(
                sum(final_accs_post) / max(len(final_accs_post), 1)
            )
            final_line_correct = (
                per_case_records[-1]["final_draft_line_correctness"]
                if per_case_records
                else {}
            )
            final_line_correct_post = (
                per_case_records[-1].get("final_draft_line_correctness_post_revise", {})
                if per_case_records
                else {}
            )
            round_label = "dual_role_swap" if bool(getattr(args, "swap_roles", False)) else "dual_fixed_roles"
            rec = {
                "step": gstep,
                "local_step": step,
                "utc": datetime.now(timezone.utc).isoformat(),
                "round": round_label,
                "case_dir": str(primary_case_dir.resolve()),
                "case_dirs": [pc["case_dir"] for pc in per_case_records],
                "n_cases_in_step": len(per_case_records),
                "loss_total": float(lf + lv),
                "loss_filer_term": float(lf),
                "loss_verifier_term": float(lv),
                "filer_reward_mean": float(
                    sum(filer_rewards_step) / max(len(filer_rewards_step), 1)
                ),
                "verifier_reward_mean": float(
                    sum(verifier_rewards_step) / max(len(verifier_rewards_step), 1)
                ),
                "approver_reward_mean": float(
                    sum(approver_rewards_step) / max(len(approver_rewards_step), 1)
                ) if approver_rewards_step else 0.0,
                "filer_rewards": filer_rewards_step,
                "verifier_rewards": verifier_rewards_step,
                "approver_rewards": approver_rewards_step,
                "final_draft_accuracy_strict": final_acc,
                "final_draft_accuracy_per_case": final_accs,
                "final_draft_accuracy_post_revise": final_acc_post,
                "final_draft_accuracy_post_revise_per_case": final_accs_post,
                "skipped_optimizer": False,
                "notes": None,
                "gradient_accumulation_order": "paired_actions",
                "role_assignment": roles,
                "transitions_added": len(transitions),
                "buffer_size": len(buffer),
                "replay_disabled": replay_disabled,
                "replay_batch_size": replay_batch_size,
                "replay_updates_used": replay_used,
                "on_policy_updates_used": len(per_case_records),
                "case_histogram": case_hist,
                "sac_rate": sac_rate,
                "invalid_filer_json_rate": invalid_filer_json_rate,
                "revision_considered_count": int(n_revise_considered),
                "invalid_revise_json_rate": invalid_revise_json_rate,
                "revise_rate": revise_rate,
                "invalid_approver_json_rate": invalid_approver_json_rate,
                "decision_accuracy": decision_accuracy,
                "decision_right_count": int(n_decision_right),
                "decision_wrong_count": int(n_decision_wrong),
                "approver_forced_revise_rate": approver_forced_revise_rate,
                "approver_forced_revise_count": int(n_forced_explore),
            }
            metrics_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            metrics_fp.flush()
            buf_stats_fp.write(
                json.dumps(
                    {
                        "step": gstep,
                        "local_step": step,
                        "utc": rec["utc"],
                        "role_assignment": roles,
                        "transitions_added": len(transitions),
                        "buffer_size": len(buffer),
                        "replay_disabled": replay_disabled,
                        "recent_window_size": min(
                            len(buffer), int(args.replay_recent_window)
                        ),
                        "replay_warmup": int(args.replay_warmup),
                        "replay_updates_used": replay_used,
                        "replay_batch_size": replay_batch_size,
                        "case_histogram": case_hist,
                        "sac_rate": sac_rate,
                        "invalid_filer_json_rate": invalid_filer_json_rate,
                        "revision_considered_count": int(n_revise_considered),
                        "invalid_revise_json_rate": invalid_revise_json_rate,
                        "revise_rate": revise_rate,
                        "invalid_approver_json_rate": invalid_approver_json_rate,
                        "decision_accuracy": decision_accuracy,
                        "approver_forced_revise_rate": approver_forced_revise_rate,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            buf_stats_fp.flush()
            step_case_label = (
                primary_case_dir.name
                if len(per_case_records) <= 1
                else f"{len(per_case_records)}cases"
            )
            # rounds.tsv columns (per the header on line ~1948):
            #   global_step  round  case_dir  loss  filer_r_mean  ver_r_mean  skipped  notes
            # Earlier dual-agent code wrote final_acc into the filer_r_mean
            # column and hardcoded 0.0 into ver_r_mean — that's why sixth_run's
            # rounds.tsv shows ver_r_mean=0 across all steps even though
            # metrics.jsonl had the real verifier_reward_mean (~0.55-0.65).
            tsv_fp.write(
                f"{gstep}\t{round_label}\t{step_case_label}\t{float(lf+lv):.6f}\t"
                f"{float(rec.get('filer_reward_mean', 0.0)):.6f}\t"
                f"{float(rec.get('verifier_reward_mean', 0.0)):.6f}\t0\t"
                f"acc={final_acc:.4f} acc_post={final_acc_post:.4f} "
                f"ap_r={float(rec.get('approver_reward_mean', 0.0)):.4f}\n"
            )
            tsv_fp.flush()
            _write_step_trace(
                trace_dir,
                gstep,
                {
                    "step": gstep,
                    "utc": rec["utc"],
                    "case_dir": str(primary_case_dir.resolve()),
                    "case_dirs": [pc["case_dir"] for pc in per_case_records],
                    "n_cases_in_step": len(per_case_records),
                    "per_case": per_case_records,
                    "role_assignment": roles,
                    "transitions_added": len(transitions),
                    "buffer_size": len(buffer),
                    "replay_disabled": replay_disabled,
                    "replay_batch_size": replay_batch_size,
                    "replay_updates_used": replay_used,
                    "on_policy_updates_used": len(per_case_records),
                    "case_histogram": case_hist,
                    "sac_rate": sac_rate,
                    "invalid_filer_json_rate": invalid_filer_json_rate,
                    "revision_considered_count": int(n_revise_considered),
                    "invalid_revise_json_rate": invalid_revise_json_rate,
                    "revise_rate": revise_rate,
                    "invalid_approver_json_rate": invalid_approver_json_rate,
                    "decision_accuracy": decision_accuracy,
                    "decision_right_count": int(n_decision_right),
                    "decision_wrong_count": int(n_decision_wrong),
                    "transitions": transitions,
                    "final_draft_accuracy_strict": final_acc,
                    "final_draft_accuracy_per_case": final_accs,
                    "final_draft_line_correctness": final_line_correct,
                    "final_draft_accuracy_post_revise": final_acc_post,
                    "final_draft_accuracy_post_revise_per_case": final_accs_post,
                    "final_draft_line_correctness_post_revise": final_line_correct_post,
                },
            )
            if step % args.log_every == 0:
                forced_str = (
                    f" forced_rev={approver_forced_revise_rate:.2f}"
                    if float(getattr(args, "approver_explore_eps", 0.0) or 0.0) > 0.0
                    else ""
                )
                print(
                    f"[dual step global={gstep} local={step}/{args.steps}] "
                    f"cases={len(per_case_records)} "
                    f"loss={float(lf+lv):.4f} trans={len(transitions)} buffer={len(buffer)} "
                    f"roles A={roles['A_role']} B={roles['B_role']} "
                    f"sac={sac_rate:.2f} fbad={invalid_filer_json_rate:.2f} "
                    f"rev_bad={invalid_revise_json_rate:.2f} rev_rate={revise_rate:.2f} "
                    f"ap_bad={invalid_approver_json_rate:.2f} dec_acc={decision_accuracy:.2f}"
                    f"{forced_str} "
                    f"acc={final_acc:.3f} acc_post={final_acc_post:.3f}"
                )
            if step % args.save_every == 0 and args.use_lora:
                save_path = run_dir / f"adapter_step_{gstep}"
                _save_lora_checkpoint(
                    model, tokenizer, run_dir, save_path, gstep, optimizer, scaler
                )
                print(f"[train_grpo] Saved adapter + training_state to {save_path}")
            if int(args.eval_every) > 0 and test_cases and step % int(args.eval_every) == 0:
                eval_out = _run_dual_periodic_eval(
                    model, tokenizer, test_cases, args, device, gstep, eval_fp
                )
                # Best-checkpoint tracking + early stopping. Only fires on
                # eval boundaries (not every step), so the cost is bounded.
                # 'eval_state' is initialised lazily on the first eval and
                # carries the running best score, count of stale evals, and
                # the path to the best adapter snapshot.
                if eval_out is not None:
                    eval_state = locals().get("eval_state") or {
                        "best_score": float("-inf"),
                        "best_step": 0,
                        "best_path": None,
                        "evals_since_improvement": 0,
                        "metric_key": (
                            "test_accuracy_strict_post_revise_mean"
                            if str(getattr(args, "best_metric", "post_revise"))
                            == "post_revise"
                            else "test_accuracy_strict_mean"
                        ),
                    }
                    metric_key = eval_state["metric_key"]
                    score = float(eval_out.get(metric_key, 0.0))
                    if score > eval_state["best_score"] + 1e-12:
                        eval_state["best_score"] = score
                        eval_state["best_step"] = gstep
                        eval_state["evals_since_improvement"] = 0
                        if (
                            args.use_lora
                            and bool(getattr(args, "save_best_adapter", True))
                        ):
                            best_path = run_dir / f"best_adapter_step_{gstep}"
                            try:
                                _save_lora_checkpoint(
                                    model,
                                    tokenizer,
                                    run_dir,
                                    best_path,
                                    gstep,
                                    optimizer,
                                    scaler,
                                )
                                eval_state["best_path"] = str(best_path)
                                # Maintain a stable pointer file so external
                                # tools (paper tables, eval scripts) can find
                                # the best checkpoint without scanning dirs.
                                (run_dir / "best.json").write_text(
                                    json.dumps(
                                        {
                                            "best_step": gstep,
                                            "best_score": score,
                                            "best_metric": metric_key,
                                            "best_path": str(best_path),
                                            "all_eval_steps_so_far": int(
                                                step // int(args.eval_every)
                                            ),
                                        },
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                                print(
                                    f"[train_grpo] NEW BEST: {metric_key}={score:.4f} "
                                    f"at step {gstep}; saved to {best_path}"
                                )
                            except Exception as ee:
                                print(
                                    f"[train_grpo] Warning: failed to save best "
                                    f"adapter at step {gstep}: {ee}"
                                )
                    else:
                        eval_state["evals_since_improvement"] += 1
                        print(
                            f"[train_grpo] eval step {gstep}: {metric_key}="
                            f"{score:.4f} (no improvement; best={eval_state['best_score']:.4f} "
                            f"at step {eval_state['best_step']}; "
                            f"stale={eval_state['evals_since_improvement']})"
                        )
                    patience = int(getattr(args, "early_stop_patience", 0))
                    min_step = int(getattr(args, "early_stop_min_step", 5))
                    if (
                        patience > 0
                        and gstep >= min_step
                        and eval_state["evals_since_improvement"] >= patience
                    ):
                        print(
                            f"[train_grpo] EARLY STOP at step {gstep}: "
                            f"{eval_state['evals_since_improvement']} consecutive "
                            f"evals without improvement (patience={patience}). "
                            f"Best {metric_key}={eval_state['best_score']:.4f} "
                            f"at step {eval_state['best_step']}."
                        )
                        try:
                            (run_dir / "early_stopped.json").write_text(
                                json.dumps(
                                    {
                                        "stopped_at_step": gstep,
                                        "best_step": eval_state["best_step"],
                                        "best_score": eval_state["best_score"],
                                        "best_metric": metric_key,
                                        "patience": patience,
                                        "evals_since_improvement": eval_state[
                                            "evals_since_improvement"
                                        ],
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                        break
    finally:
        buf_stats_fp.close()
    if args.use_lora:
        final = run_dir / "adapter_final"
        g_final = global_step_base + args.steps
        _save_lora_checkpoint(model, tokenizer, run_dir, final, g_final, optimizer, scaler)
        print(f"[train_grpo] Saved final adapter + training_state (step {g_final}) to {final}")


def main() -> None:
    p = argparse.ArgumentParser(description="On-policy GRPO for Qwen filer + verifier")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["tax", "finqa", "convfinqa"],
        default="tax",
        help="Benchmark dataset: tax (TaxCalcBench), finqa, or convfinqa.",
    )
    p.add_argument(
        "--cases-root",
        type=str,
        default=None,
        help="Root directory to search recursively for case dirs (input.json + gold file). "
        "Defaults to the dataset-specific train/eval root.",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Fraction of discovered cases to hold out as test split (0 disables split).",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic train/test split.",
    )
    p.add_argument(
        "--train-case-subset-size",
        type=int,
        default=0,
        help="If >0, restrict training to a fixed subset of N train cases (taken "
        "deterministically from the sorted train split). Lower N = lower per-step "
        "variance. Use with --train-case-cycle for fully reproducible per-step order.",
    )
    p.add_argument(
        "--train-case-cycle",
        action="store_true",
        default=False,
        help="Iterate through the (subset of) train cases deterministically in sorted "
        "order using (step-1) %% len(train_cases) instead of sampling randomly. "
        "Combined with --train-case-subset-size this makes each step see a fully "
        "predetermined case → much lower variance across steps.",
    )
    p.add_argument(
        "--train-all-cases-per-step",
        action="store_true",
        default=False,
        help="Within each step, iterate over ALL cases in the (subset of) train pool "
        "and perform one on-policy update per case. With --train-case-subset-size 3 "
        "and --steps 30 this yields 90 on-policy updates. Replay still runs once at "
        "the end of the step using the combined buffer.",
    )
    p.add_argument(
        "--train-case-resample-each-step",
        action="store_true",
        default=False,
        help="Per-step random sub-sampling from the FULL train pool. When set, each "
        "step draws a fresh --train-case-subset-size sized random subset from the "
        "complete --train split (instead of cycling the same first-N cases forever). "
        "Diagnosed in tenth_run analysis: cycling 3-5 of 41 train cases caused the "
        "model to memorise the subset by step ~5 and stop improving (best post-revise "
        "= 0.456 at step 2 of 30). Resampling restores gradient diversity. Implies "
        "--train-all-cases-per-step semantics for the drawn subset, but does NOT use "
        "all 41 cases per step (that would be ~14x slower).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "TAX_MODEL_PATH", "/home/ubuntu/models/models--Qwen--Qwen3-4B"
        ),
    )
    p.add_argument("--output-dir", type=str, default="grpo_checkpoints")
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Subfolder under output-dir for logs and checkpoints; default UTC timestamp",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="K rollouts per GRPO group (>=2). Use 2 if CUDA OOM",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta-kl", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--filer-max-new-tokens",
        type=int,
        default=4096,
        help="Lower this if CUDA OOM during training (draft JSON length cap)",
    )
    p.add_argument(
        "--verifier-max-new-tokens",
        type=int,
        default=3072,
        help="Lower if OOM on verifier GRPO backward",
    )
    p.add_argument("--invalid-json-reward", type=float, default=-0.5)
    p.add_argument(
        "--revise-format-penalty",
        type=float,
        default=0.25,
        help="When a revision is considered but the filer produces invalid JSON, use "
        "revise_reward = keep_reward - this_penalty (instead of a hard invalid_json_reward). "
        "This keeps a format signal without making REVISE dominated by parse noise.",
    )
    p.add_argument(
        "--skip-verifier-update-on-invalid-filer",
        action="store_true",
        default=True,
        help="Skip the verifier paired-action update on transitions where the filer line "
        "could not be parsed; otherwise those rows inject an artificial 'wrong' label and "
        "reward SAC-always behavior.",
    )
    p.add_argument(
        "--skip-filer-revise-update-on-invalid-revise",
        action="store_true",
        default=True,
        help="Skip the filer paired-action KEEP/REVISE update when the REVISE output was "
        "invalid JSON. The format signal is already reflected in revise_reward via "
        "--revise-format-penalty; skipping prevents the gradient from chasing a noisy text.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Previous run directory containing adapter_final/; loads LoRA + optional training_state.pt",
    )
    p.add_argument("--use-lora", action="store_true", help="Train LoRA adapters (recommended)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast (bf16)")
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade speed for lower VRAM (recommended if OOM)",
    )
    p.add_argument(
        "--alternate-rounds",
        action="store_true",
        help="Each step still trains filer+verifier; alternate which GRPO backward runs first "
        "(filer vs verifier) on odd/even local steps (see --round-start)",
    )
    p.add_argument(
        "--round-start",
        type=str,
        choices=("filer", "verifier"),
        default="filer",
        help="With --alternate-rounds: if 'filer', odd steps accumulate filer grads first; "
        "if 'verifier', odd steps accumulate verifier grads first",
    )
    p.add_argument(
        "--case-taxonomy",
        action="store_true",
        help="Scale filer/verifier GRPO losses from tax_case_taxonomy (best verifier sample)",
    )
    p.add_argument(
        "--case-x-good",
        type=float,
        default=CaseTaxonomyConfig.x_good,
        help="Threshold on reward_filer_draft for 'good draft' x (case classifier)",
    )
    p.add_argument(
        "--case-verifier-good",
        type=float,
        default=CaseTaxonomyConfig.verifier_good,
        help="Threshold on verifier rewards for SAC/full parse (case classifier)",
    )
    p.add_argument(
        "--case-yprime-margin",
        type=float,
        default=CaseTaxonomyConfig.y_prime_gt_y_margin,
        help="Min (r_full - r_sac) to treat y' as strictly better than y",
    )
    p.add_argument(
        "--no-case-taxonomy-revise",
        action="store_true",
        help="With --case-taxonomy: do not run extra filer revise sample for z / keep-x classification",
    )
    p.add_argument(
        "--case-revise-draft-amount-abs-tol",
        type=float,
        default=1e-2,
        help="Line amount tolerance for drafts_near_identical (revise_chose_keep_x)",
    )
    p.add_argument(
        "--line-case-taxonomy",
        action="store_true",
        help="Per-line 6-case labels (y=SAC, y'=full from same JSON); boost GRPO for cases 1&5 vs others",
    )
    p.add_argument(
        "--line-case-reward-scale",
        type=float,
        default=0.15,
        help="Add scale * case_preference_score to each filer/verifier reward before advantage norm",
    )
    p.add_argument(
        "--line-case-prefer-boost",
        type=float,
        default=0.15,
        help="Extra multiplier on wf,wv when case mix favors 1&5 over 2–4&6",
    )
    p.add_argument(
        "--line-case-z-improve-eps",
        type=float,
        default=1e-4,
        help="Reserved: global z vs x improvement (per-line z/z′ use line amounts)",
    )
    p.add_argument(
        "--line-case-amount-tol",
        type=float,
        default=1e-6,
        help="Per-line |amount_z − amount_x| above this counts as revision touching the line",
    )
    p.add_argument(
        "--linewise-rollout",
        action="store_true",
        help="Generate filer/verifier per evaluated line (short JSONs) then aggregate for one GRPO step",
    )
    p.add_argument(
        "--filer-outcome-bonus",
        type=float,
        default=0.5,
        help="Alpha layered on top of the counterfactual decision reward: "
        "keep_reward += alpha * S_x and revise_reward += alpha * S_z. 0 disables. "
        "Adds a numerical-success signal on top of the always-non-zero counterfactual "
        "decision gradient.",
    )
    p.add_argument(
        "--verifier-loss-scale",
        type=float,
        default=0.5,
        help="Scale applied to the verifier-side GRPO loss during gradient accumulation. "
        "Bumped from 0.25 -> 0.5 after fifth_run showed the verifier going silent "
        "(sac_rate drift 0.26 -> 0.17, cases 4 dominant).",
    )
    p.add_argument(
        "--revise-sample-k",
        type=int,
        default=5,
        help="Number of revise candidates the filer samples per line; the best one "
        "(by S_z=1, then parse-validity, then first) is used for the Approver "
        "prompt and the rollout's revise reward. K=1 keeps the legacy single-sample "
        "behaviour. Bumped from 3 -> 5 after sixth_run showed many lines with "
        "Sz=0/9 across 9 steps x 3 candidates (e.g. case 0 line 34 SAC 8/9 / Sz 0/9). "
        "K=5 buys more exploration without going so wide the budget explodes.",
    )
    p.add_argument(
        "--revise-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used specifically for the revise step. Decoupled "
        "from --temperature so we can keep deterministic-ish drafts (temp 0.2) while "
        "exploring more diverse revisions. Bumped from 0.5 -> 0.7 in concert with "
        "--revise-sample-k=5 to widen the candidate distribution; mode-collapsed "
        "revises were a sixth_run failure mode for the calculation cascade lines.",
    )
    p.add_argument(
        "--approver-cave-penalty",
        type=float,
        default=0.2,
        help="Negative reward applied to approver_revise_reward when the Approver "
        "REVISEs a correct draft (S_x=1) AND the revise sample fails to land gold "
        "(S_z=0). Sixth_run case 1 line 1a had Sx=9/9 but Approver REVISE 4/9 "
        "with Sz=5/9 — KEEP-vs-REVISE gap of ~1.5 wasn't a strong enough signal. "
        "Bumped 0 -> 0.5 for seventh_run, which over-corrected: Approver "
        "collapsed to ~always-KEEP (revise_rate 0.07 by step 13, case 5a -> 0). "
        "Lowered to 0.2 so KEEP-vs-REVISE gap stays asymmetric (~1.7) without "
        "making KEEP a degenerate dominant strategy. Set to 0 to fall back to "
        "the symmetric outcome-bonus reward.",
    )
    p.add_argument(
        "--approver-target-revise-rate",
        type=float,
        default=0.30,
        help="Target REVISE rate (over Approver-considered transitions) used by the "
        "homeostatic balance term. When the step's observed revise-rate falls below "
        "this target, a per-step bonus = balance_strength * (target - observed) is "
        "added to every approver_revise_reward in the step before computing GRPO "
        "advantages. Calibrated from observed case-mix on tenth_run step 1 (~14 "
        "considered transitions, of which ~25-40%% are right to revise).",
    )
    p.add_argument(
        "--approver-balance-strength",
        type=float,
        default=0.0,
        help="Strength of the homeostatic Approver-balance bonus. 0 disables (default, "
        "preserves prior behavior). Recommended 0.5-1.0. Diagnosed in tenth_run: from "
        "step 6, revise_rate dropped 0.43 -> 0.12 and post_revise = strict (multi-agent "
        "loop became no-op). Same collapse pattern was observed in seventh_run with "
        "approver_cave_penalty=0.5. The cave penalty is asymmetric (only fires on "
        "false-positive REVISE) and so leaves a 'always KEEP' local optimum once the "
        "Approver learns it. This term is a homeostatic counter-pressure: when revise "
        "actions get too rare, REVISE rewards get a bonus. Acts only at the step "
        "level, costs nothing in expectation when revise_rate >= target.",
    )
    p.add_argument(
        "--approver-rare-case-weight",
        type=float,
        default=0.0,
        help="Inverse-frequency loss weighting on the Approver gradient. 0 disables "
        "(default, preserves prior behavior). When > 0, each Approver transition's "
        "gradient is scaled by (avg_count / count(case_id)) ** alpha where alpha is "
        "this flag. alpha=1.0 = full inverse-frequency, alpha=0.5 = sqrt-inverse. "
        "Rationale: in a typical step the case mix is dominated by case-1 (filer "
        "right, no SAC, ~50%% of lines) and case-4 (filer wrong, V silent, ~25%%); "
        "rare wins like case-5a (~1-2%% of lines) get drowned out. Inverse-frequency "
        "weighting amplifies the gradient on rare-but-critical transitions.",
    )
    p.add_argument(
        "--approver-explore-eps",
        type=float,
        default=0.0,
        help="Forced-exploration probability for the Approver during *training* "
        "rollouts. With probability eps, override the Approver's sampled "
        "KEEP/REVISE decision with REVISE on every V-SAC transition. The "
        "paired-action GRPO loss is unchanged (it always trains on both KEEP "
        "and REVISE counterfactually), but the rollout's case_histogram and "
        "case_counts shift: case-5a / 5b transitions surface that would "
        "otherwise have been hidden as 6a / 6b. Combined with "
        "--approver-rare-case-weight > 0 this amplifies the gradient signal "
        "on the rare 'REVISE was right' transitions that the Approver never "
        "naturally explores once it collapses to always-KEEP "
        "(observed in 4b_5/4b_10/4b_10_v2/8b_3 by step ~5). 0.0 disables "
        "(default). Recommended 0.3-0.5. Eval-time rollouts are NOT affected "
        "(the Approver is sampled normally for test metrics).",
    )
    p.add_argument(
        "--dual-agent-replay",
        action="store_true",
        help="Use two adapters (phi/theta), role swap, line-level transition buffer, and replay updates",
    )
    p.add_argument(
        "--swap-roles",
        action="store_true",
        default=False,
        help="Enable periodic role swapping between filer and verifier adapters. "
        "Default: OFF (filer always trains on 'default' adapter, verifier on 'theta'). "
        "Swapping was the default in fifth_run and earlier; it disrupted Approver "
        "and Verifier learning, so it's now opt-in for explicit symmetry experiments. "
        "When enabled, the cadence is controlled by --role-swap-period.",
    )
    p.add_argument("--role-swap-period", type=int, default=10)
    p.add_argument("--replay-capacity", type=int, default=200000)
    p.add_argument("--replay-recent-window", type=int, default=20000)
    p.add_argument("--replay-warmup", type=int, default=5000)
    p.add_argument("--replay-batch", type=int, default=512)
    p.add_argument("--replay-updates-per-iter", type=int, default=1)
    p.add_argument(
        "--disable-replay",
        action="store_true",
        default=False,
        help="Disable the off-policy replay buffer in dual-agent mode. When set, "
        "transitions are not stored for replay and no replay updates are performed. "
        "The on-policy paired-action GRPO update still runs for every new transition.",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run periodic evaluation every N train steps (0 disables).",
    )
    # ------------------------------------------------------------------
    # Best-checkpoint tracking + early stopping (added after 8b_3 lost the
    # 0.572 step-10 peak when later steps regressed to 0.526). When enabled
    # the trainer remembers the best test_accuracy_strict_post_revise_mean
    # observed so far and saves the corresponding adapter under
    #   <run_dir>/best_adapter_step_<gstep>/
    # writing a <run_dir>/best.json pointer file. Optional early-stopping
    # halts training when N consecutive evals fail to improve over the best
    # observed value, saving the GPU-hr that 8b_3 burned past step 10.
    # ------------------------------------------------------------------
    p.add_argument(
        "--save-best-adapter",
        action="store_true",
        default=True,
        help="After each periodic eval, if test_accuracy_strict_post_revise_mean "
        "improved over the running best, copy the current adapter(s) to "
        "<run_dir>/best_adapter_step_<gstep>/ and update <run_dir>/best.json. "
        "Default ON; pass --no-save-best-adapter to disable.",
    )
    p.add_argument(
        "--no-save-best-adapter",
        dest="save_best_adapter",
        action="store_false",
        help="Disable best-checkpoint saving (overrides --save-best-adapter).",
    )
    p.add_argument(
        "--best-metric",
        type=str,
        default="post_revise",
        choices=("post_revise", "strict"),
        help="Which test metric to track for best-checkpoint and early-stop. "
        "post_revise = test_accuracy_strict_post_revise_mean (full multi-agent "
        "loop on test set, default). strict = test_accuracy_strict_mean "
        "(filer-only). Use 'strict' when running with --no-multi-agent.",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="If >0, stop training when this many consecutive periodic evals "
        "fail to improve over the running best. 0 disables (default). "
        "Recommended: 2-3 for stability. 8b_3 would have stopped at step 15 "
        "(saving ~10 GPU-hr) since post_revise peaked at step 10 and didn't "
        "recover by step 15 or 20.",
    )
    p.add_argument(
        "--early-stop-min-step",
        type=int,
        default=5,
        help="Minimum global step before early stopping can trigger. Prevents "
        "stopping during the noisy warm-up phase. Default 5 = always run at "
        "least the first eval.",
    )
    # ------------------------------------------------------------------
    # Filer-only baseline mode. Trains the Filer adapter via paired-action
    # GRPO on K=2 sampled drafts per line, with reward = oracle line
    # correctness S_x. Skips the Verifier and Approver entirely. Provides
    # the missing "trained-Filer-only" baseline for Table 1 isolating the
    # contribution of the multi-agent loop.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-multi-agent",
        action="store_true",
        default=False,
        help="Filer-only baseline: skip Verifier and Approver entirely. Per "
        "evaluated line, sample K=2 filer drafts, reward each by S_x (oracle "
        "line correctness), train the Filer adapter via paired-action GRPO. "
        "Eval still uses _dual_linewise_rollout_transitions but with V/A "
        "shortcircuited (post_revise == strict). Provides the missing "
        "trained-Filer-only control row for the paper's main results table.",
    )
    # ------------------------------------------------------------------
    # PPO-clip baselines for the Filer-only branch: DAPO and GSPO. Both
    # require --no-multi-agent (PPO-clip dispatch is only wired into the
    # filer-only update path). Mutually exclusive with each other and with
    # the default REINFORCE-style GRPO loss. KL anchor (--beta-kl) is
    # automatically zeroed when either flag is set so the clip controls
    # policy drift, matching the algorithms' published formulations.
    # ------------------------------------------------------------------
    p.add_argument(
        "--use-dapo",
        action="store_true",
        default=False,
        help="Filer-only baseline using the DAPO (Decoupled Clip + Dynamic "
        "Sampling) PPO variant from Yu et al. 2025 (ByteDance Seed + "
        "Tsinghua AIR) instead of REINFORCE-style GRPO. Asymmetric clip "
        "(--dapo-clip-low, --dapo-clip-high), token-level loss reduction "
        "(mean over completion tokens within each sequence), and dynamic "
        "sampling (skip groups whose K rollouts share the same reward, "
        "already enforced by the std>1e-12 guard in the filer-only "
        "update). Requires --no-multi-agent. Mutually exclusive with "
        "--use-gspo. Sets the effective KL weight to 0 inside the clip "
        "loss.",
    )
    p.add_argument(
        "--dapo-clip-low",
        type=float,
        default=0.2,
        help="Lower clip bound (1 - epsilon_low) for DAPO. Paper default "
        "0.2. Only used when --use-dapo is set.",
    )
    p.add_argument(
        "--dapo-clip-high",
        type=float,
        default=0.28,
        help="Upper clip bound (1 + epsilon_high) for DAPO. Paper default "
        "0.28; the asymmetry (high > low) is the Decoupled Clip - it "
        "permits larger upweighting of low-probability good samples "
        "while still bounding aggressive over-confidence. Only used "
        "when --use-dapo is set.",
    )
    p.add_argument(
        "--use-gspo",
        action="store_true",
        default=False,
        help="Filer-only baseline using GSPO (Group Sequence Policy "
        "Optimization, Zheng et al. 2025, Alibaba/Qwen) instead of "
        "REINFORCE-style GRPO. Sequence-level importance ratio - "
        "ratio_seq = exp(mean_t(lp_new_t - lp_old_t)) (geometric mean "
        "of per-token ratios) - and a single symmetric clip (--gspo-clip). "
        "GSPO's claim is that token-level ratios produce high-variance "
        "updates on long sequences (and especially on MoE models); "
        "sequence-level ratios stabilise this. Requires --no-multi-agent. "
        "Mutually exclusive with --use-dapo. Sets the effective KL "
        "weight to 0 inside the clip loss.",
    )
    p.add_argument(
        "--gspo-clip",
        type=float,
        default=0.2,
        help="Symmetric clip bound for GSPO (1 +/- this value). Paper "
        "default 0.2 (matching vanilla PPO). Only used when --use-gspo "
        "is set.",
    )
    # ------------------------------------------------------------------
    # Approver missed-opportunity penalty (counterpart to --approver-cave-
    # penalty). Fires on case 6b: filer wrong, V SAC, S_z=1 (good revise
    # exists), Approver KEEP. Pushes the Approver harder away from the
    # "always-KEEP" attractor when a fixable error was on the table.
    # ------------------------------------------------------------------
    p.add_argument(
        "--approver-miss-penalty",
        type=float,
        default=0.0,
        help="Negative reward applied to approver_keep_reward when the Approver "
        "KEEPs a wrong draft (S_x=0) AND a good revise candidate exists "
        "(S_z=1) — the case-6b 'missed easy fix' situation. Default 0.0 "
        "(disabled). Recommended 0.3-0.5. Symmetric counterpart to "
        "--approver-cave-penalty: cave penalises false-positive REVISE, "
        "miss penalises false-negative KEEP. Together they widen the "
        "REVISE-vs-KEEP gradient gap on both sides of the decision boundary, "
        "fighting the always-KEEP collapse seen in 4b_5/4b_10/8b_3.",
    )
    p.add_argument(
        "--eval-max-cases",
        type=int,
        default=8,
        help="Max number of test cases to evaluate per periodic eval (<=0 means all).",
    )
    p.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for periodic eval rollouts.",
    )
    p.add_argument(
        "--eval-top-p",
        type=float,
        default=1.0,
        help="Top-p for periodic eval rollouts.",
    )
    args = p.parse_args()
    configure_dataset(args.dataset)
    if not args.cases_root:
        args.cases_root = str(get_dataset_config(args.dataset).default_cases_root)
    if getattr(args, "line_case_taxonomy", False):
        args.case_taxonomy = True
    if getattr(args, "dual_agent_replay", False):
        args.linewise_rollout = True
        if not args.use_lora:
            raise SystemExit("--dual-agent-replay requires --use-lora")
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 for GRPO normalization")
    use_dapo = bool(getattr(args, "use_dapo", False))
    use_gspo = bool(getattr(args, "use_gspo", False))
    if use_dapo and use_gspo:
        raise SystemExit(
            "--use-dapo and --use-gspo are mutually exclusive (they both replace "
            "the GRPO loss). Pick one."
        )
    if (use_dapo or use_gspo) and not bool(getattr(args, "no_multi_agent", False)):
        raise SystemExit(
            "--use-dapo / --use-gspo are only wired into the Filer-only update "
            "path; pass --no-multi-agent to use them as single-agent baselines."
        )
    train_loop(args)


if __name__ == "__main__":
    main()
