#!/usr/bin/env python3
"""DPA-GRPO: paired-action GRPO for a Filer / Verifier / Approver triple sharing one Qwen backbone.

Algorithm sketch (Algorithm 1 in the paper):
  For each training step:
    For each line l of the 19 evaluated 1040 lines:
      x_l   ~ pi_phi(. | input, context_<l)         # filer line draft
      y_l   ~ pi_theta(. | input, x_l)              # verifier {SAC, NS}
      if y_l == SAC:
        z_l = best_of_K from pi_phi(. | input, x_l, y_l)    # filer revise
        a_l ~ pi_phi_approver(. | x_l, y_l, z_l)             # {KEEP, REVISE}
      build counterfactual paired-action rewards over {SAC,NS}, {KEEP,REVISE}
    GRPO update: -A_i * sum_t log pi(a_t) + beta * KL(pi || pi_ref)

The filer / verifier / approver are three LoRA adapters on the same backbone:
  filer_adapter    = "default"   (filer drafts and revises live here)
  verifier_adapter = "theta"     (verifier action stub trained here)
  approver_adapter = "default"   (KEEP/REVISE token trained as a head of the filer adapter)
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

from peft import LoraConfig, PeftModel, get_peft_model

from device_pick import resolve_torch_device
from agents import (
    _trim_draft_for_verifier,
    _trim_input_for_verifier,
    parse_verifier_raw_to_safety_case,
)
from client import extract_json_from_response
from evaluator import evaluated_line_ids, line_strict_correctness_by_line_id
from prompts import (
    APPROVER_LINE_SYSTEM_PROMPT,
    FILER_LINE_SYSTEM_PROMPT,
    VERIFIER_LINE_SYSTEM_PROMPT,
)
from rewards import reward_filer_draft
from schemas import DraftLine, DraftReturn
from tax_case_taxonomy import CaseTaxonomyConfig
from tax_line_case import (
    LineCaseConfig,
    case_preference_score,
    classify_line_case_id,
    line_case_loss_weights,
)


# Fixed three-adapter routing. Filer drafts and revises share the "default"
# adapter; the verifier stub trains on a second LoRA "theta"; the approver
# decision token is a head of the filer adapter and so trains on "default".
_ROLES: Dict[str, str] = {
    "filer_adapter": "default",
    "verifier_adapter": "theta",
    "approver_adapter": "default",
}


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def _save_training_state(
    run_dir: Path, last_step: int, optimizer: AdamW, scaler: Optional[Any],
) -> None:
    payload: Dict[str, Any] = {"last_step": last_step, "optimizer": optimizer.state_dict()}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, run_dir / "training_state.pt")


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
    has_weight = (save_dir / "adapter_model.safetensors").exists() or (
        save_dir / "adapter_model.bin"
    ).exists()
    if not has_weight or not (save_dir / "adapter_config.json").exists():
        raise RuntimeError(f"LoRA artifacts missing in {save_dir}")
    _save_training_state(run_dir, step_id, optimizer, scaler)


def _resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if (p / "snapshots").exists():
        snaps = list((p / "snapshots").iterdir())
        if snaps:
            return str(snaps[0])
    return str(p)


# -----------------------------------------------------------------------------
# Case discovery and train/test split
# -----------------------------------------------------------------------------

def discover_case_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for input_path in root.rglob("input.json"):
        d = input_path.parent
        if (d / "output.xml").exists():
            out.append(d)
    return sorted(set(out))


def split_train_test_cases(
    cases: List[Path], test_fraction: float, seed: int,
) -> Tuple[List[Path], List[Path]]:
    if not cases:
        return [], []
    frac = max(0.0, min(0.95, float(test_fraction)))
    if frac <= 0.0 or len(cases) < 2:
        return list(cases), []
    n_test = max(1, min(len(cases) - 1, int(round(len(cases) * frac))))
    ordered = sorted(cases, key=lambda p: str(p.resolve()))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    test_cases = sorted(ordered[:n_test], key=lambda p: str(p.resolve()))
    train_cases = sorted(ordered[n_test:], key=lambda p: str(p.resolve()))
    return train_cases, test_cases


def _select_case_for_step(
    cases: List[Path], step: int, cycle: bool, rng: random.Random,
) -> Path:
    if not cases:
        raise ValueError("empty case list")
    if cycle:
        return cases[(max(int(step), 1) - 1) % len(cases)]
    return rng.choice(cases)


def load_case_input(case_dir: Path) -> Dict[str, Any]:
    with (case_dir / "input.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("input", data)


# -----------------------------------------------------------------------------
# Per-step trace helpers
# -----------------------------------------------------------------------------

def _draft_to_dict(d: Optional[DraftReturn]) -> Optional[Dict[str, Any]]:
    if d is None:
        return None
    return {
        "return_version": d.return_version,
        "lines": [
            {"form": x.form, "line": x.line, "description": x.description,
             "amount": x.amount, "rationale": x.rationale}
            for x in d.lines
        ],
        "metadata": d.metadata,
    }


def _write_step_trace(trace_dir: Path, gstep: int, payload: Dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"step_{gstep:05d}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Tokenizer / model plumbing
# -----------------------------------------------------------------------------

def _lora_target_modules(model: torch.nn.Module) -> List[str]:
    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found: set[str] = set()
    for n, _ in model.named_modules():
        for tail in names:
            if n.endswith(tail):
                found.add(tail)
    return sorted(found)


def build_chat_text(tokenizer: Any, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    # Qwen3 chat template's enable_thinking=True auto-injects a <think> block;
    # the model then exhausts the token budget on chain-of-thought before
    # emitting JSON. Disable it so the model is forced to emit JSON directly.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# -----------------------------------------------------------------------------
# Per-line user payloads
# -----------------------------------------------------------------------------

def _line_dict(line_obj: DraftLine) -> Dict[str, Any]:
    return {
        "form": line_obj.form, "line": line_obj.line,
        "description": line_obj.description, "amount": line_obj.amount,
        "rationale": line_obj.rationale,
    }


_LINE_OUTPUT_SCHEMA = {
    "form": "1040", "line": "<target_line>", "description": "string",
    "amount": 0, "rationale": "short string",
}


def _line_filer_user_payload(
    input_json: Dict[str, Any], line_id: str, context: List[Dict[str, Any]],
) -> str:
    schema = dict(_LINE_OUTPUT_SCHEMA, line=line_id)
    return json.dumps({
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context,
        "output_schema": schema,
        "instruction": "Output ONLY one JSON object for the target line.",
    }, indent=2)


def _line_verifier_user_payload(
    input_json: Dict[str, Any], line_obj: DraftLine, line_id: str,
    context: List[Dict[str, Any]],
) -> str:
    return json.dumps({
        "input": _trim_input_for_verifier(input_json),
        "target_line": line_id,
        "context_so_far": context,
        "draft_line": _line_dict(line_obj),
        "instruction": (
            "Output ONLY JSON with one line finding for target_line using verdict in "
            "{correct, plausible, suspicious, wrong}."
        ),
    }, indent=2)


def _line_revise_user_payload(
    input_json: Dict[str, Any], line_obj: DraftLine, line_id: str,
    verifier_raw: Dict[str, Any], context: List[Dict[str, Any]],
) -> str:
    schema = dict(_LINE_OUTPUT_SCHEMA, line=line_id)
    return json.dumps({
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context,
        "draft_line": _line_dict(line_obj),
        "verifier_feedback": verifier_raw,
        "output_schema": schema,
        "instruction": "Revise only this target line and output ONLY one JSON object.",
    }, indent=2)


def _line_approver_user_payload(
    input_json: Dict[str, Any], line_obj: DraftLine, line_id: str,
    verifier_raw: Dict[str, Any], revised_line: Optional[DraftLine],
    context: List[Dict[str, Any]],
) -> str:
    return json.dumps({
        "input": input_json,
        "target_line": line_id,
        "context_so_far": context,
        "draft_line": _line_dict(line_obj),
        "verifier_feedback": verifier_raw,
        "revised_line": _line_dict(revised_line) if revised_line is not None else None,
        "instruction": (
            "Decide KEEP or REVISE for target_line and output ONLY the JSON "
            "object {\"target_line\": \"...\", \"decision\": \"KEEP\"|\"REVISE\"}."
        ),
    }, indent=2)


# -----------------------------------------------------------------------------
# JSON parsing helpers
# -----------------------------------------------------------------------------

def try_parse_line_draft(text: str, target_line: str) -> Optional[DraftLine]:
    data = extract_json_from_response((text or "").strip())
    if not data:
        return None
    if isinstance(data.get("lines"), list):
        for item in data["lines"]:
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


def _parse_approver_decision(text: str) -> Optional[str]:
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


# -----------------------------------------------------------------------------
# GRPO group update (paired-action variant: K=2 actions per group)
# -----------------------------------------------------------------------------

def completion_logprob_sum(
    model: torch.nn.Module, input_ids: torch.Tensor, prompt_length: int,
    *, enable_grad: bool = True,
) -> torch.Tensor:
    """Sum of log p over the generated continuation. enable_grad=False is for the reference forward."""
    if input_ids.dim() != 2:
        raise ValueError("expected input_ids [1, L]")
    input_ids = input_ids.long()
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    seq_len = int(input_ids.size(1))
    if max_pos is not None and seq_len > max_pos:
        raise ValueError(
            f"Sequence length {seq_len} exceeds model max_position_embeddings ({max_pos}); "
            "lower --filer-max-new-tokens / --verifier-max-new-tokens."
        )
    ctx = torch.enable_grad() if enable_grad else torch.inference_mode()
    with ctx:
        out = model(input_ids)
        log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        ti = targets.clamp(min=0, max=out.logits.size(-1) - 1)
        token_lp = log_probs.gather(-1, ti.unsqueeze(-1)).squeeze(-1)
        if prompt_length <= 0 or prompt_length >= input_ids.size(1):
            return token_lp.sum() * 0.0
        return token_lp[:, prompt_length - 1:].sum()


def _ref_context_manager(model: torch.nn.Module) -> Optional[Callable[[], Any]]:
    if isinstance(model, PeftModel):
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
    """One GRPO group: backward per sequence so peak memory is O(1) seq graph, not O(K)."""
    ref_ctx = _ref_context_manager(model)
    model.train()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_on = bool(use_amp and device_type == "cuda")
    k = max(len(sequences), 1)
    total = 0.0
    with torch.amp.autocast(device_type=device_type, enabled=amp_on, dtype=torch.bfloat16):
        for i, (seq, plen) in enumerate(zip(sequences, prompt_lengths)):
            adv_f = float(advantages[i].item())
            lp = completion_logprob_sum(model, seq, plen, enable_grad=True)
            if ref_ctx is not None:
                with ref_ctx():
                    ref_lp = completion_logprob_sum(model, seq, plen, enable_grad=False)
            else:
                ref_lp = lp.detach()
            loss = -adv_f * lp + beta_kl * (lp - ref_lp)
            total += float(loss.detach().cpu()) / k * loss_scale
            scaled = loss / k * float(loss_scale)
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
    return total


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
) -> List[torch.Tensor]:
    """Sample ``group_size`` continuations of the prompt. Bans <think> emission."""
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bad_words_ids: List[List[int]] = []
    for w in ("<think>", "</think>", "<tool_response>", "</tool_response>"):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    out: List[torch.Tensor] = []
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
            out.append(gen)
    return out


def _set_active_adapter(model: torch.nn.Module, adapter_name: str) -> None:
    if isinstance(model, PeftModel):
        model.set_adapter(adapter_name)


def _build_seq_from_prompt_and_action(
    tokenizer: Any, system_prompt: str, user_text: str, action_text: str, device: str,
) -> Tuple[torch.Tensor, int]:
    prompt_text = build_chat_text(tokenizer, system_prompt, user_text)
    p = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(p["input_ids"].shape[1])
    full = tokenizer(prompt_text + action_text, return_tensors="pt", add_special_tokens=False)
    return full["input_ids"].to(device), prompt_len


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
    seqs = generate_group_sequences(
        model, tokenizer, pid, mask, 1, max_new_tokens,
        float(args.temperature if temperature is None else temperature),
        float(args.top_p if top_p is None else top_p),
        device,
    )
    return tokenizer.decode(seqs[0][0, plen:], skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Linewise rollout: one transition per (case, line)
# -----------------------------------------------------------------------------

def _dual_linewise_rollout_transitions(
    model: torch.nn.Module,
    tokenizer: Any,
    input_json: Dict[str, Any],
    xml_path: Path,
    args: argparse.Namespace,
    device: str,
    iter_id: int,
) -> Tuple[List[Dict[str, Any]], DraftReturn, DraftReturn]:
    """Per evaluated line: filer draft -> verifier action -> if SAC: best-of-K revise + approver decision.

    Builds the counterfactual paired-action rewards (KEEP/REVISE for filer-revise and approver,
    SAC/NS for verifier) used by ``_apply_transition_batch_update`` for the GRPO update.
    """
    line_ids = list(evaluated_line_ids())
    context_so_far: List[Dict[str, Any]] = []
    selected_lines: List[DraftLine] = []
    post_revise_lines: List[DraftLine] = []
    transitions: List[Dict[str, Any]] = []

    outcome_bonus = float(args.filer_outcome_bonus)
    fmt_pen = float(args.revise_format_penalty)
    cave_penalty = float(args.approver_cave_penalty)
    miss_penalty = float(args.approver_miss_penalty)
    revise_k = max(int(args.revise_sample_k), 1)
    revise_temp = float(args.revise_temperature)
    explore_eps = float(args.approver_explore_eps)

    for lid in line_ids:
        _set_active_adapter(model, _ROLES["filer_adapter"])
        f_user = _line_filer_user_payload(input_json, lid, context_so_far)
        f_text = _sample_line_action(
            model, tokenizer, FILER_LINE_SYSTEM_PROMPT, f_user,
            args.filer_max_new_tokens, args, device,
        )
        x_parsed = try_parse_line_draft(f_text, lid)
        filer_parsed = x_parsed is not None
        x_line = x_parsed or DraftLine(
            form="1040", line=lid, description=f"Line {lid}",
            amount=0.0, rationale="invalid_json",
        )
        selected_lines.append(x_line)
        context_so_far.append(_line_dict(x_line))

        rx = bool(line_strict_correctness_by_line_id(
            DraftReturn(return_version="ty24-v1", lines=[x_line], metadata={}), xml_path,
        ).get(lid, False))
        S_x = 1.0 if rx else 0.0
        sac_correct_line = not rx

        _set_active_adapter(model, _ROLES["verifier_adapter"])
        v_user = _line_verifier_user_payload(input_json, x_line, lid, context_so_far[:-1])
        y_text = _sample_line_action(
            model, tokenizer, VERIFIER_LINE_SYSTEM_PROMPT, v_user,
            args.verifier_max_new_tokens, args, device,
        )
        y_raw = extract_json_from_response(y_text.strip()) or {}
        verdict = _extract_line_verdict_from_raw(y_raw, lid)
        y_choice = "SAC" if verdict != "correct" else "NO_SAC"
        verifier_pair_rewards = {
            "SAC": 1.0 if sac_correct_line else 0.0,
            "NO_SAC": 0.0 if sac_correct_line else 1.0,
        }

        revision_considered = y_choice == "SAC"
        revise_text = ""
        z_line: Optional[DraftLine] = None
        S_z = S_x
        keep_reward = S_x
        revise_reward = S_x
        approver_keep_reward = 0.0
        approver_revise_reward = 0.0
        revision_decision = "KEEP"
        approver_text = ""
        approver_parsed: Optional[bool] = None
        a_user = ""
        revise_parsed: Optional[bool] = None
        forced_explore = False
        r_user = ""

        if revision_considered:
            _set_active_adapter(model, _ROLES["filer_adapter"])
            r_user = _line_revise_user_payload(input_json, x_line, lid, y_raw, context_so_far[:-1])
            # Best-of-K revise: prefer (a) lands gold, (b) parses, (c) first sample.
            best_text = ""
            best_line: Optional[DraftLine] = None
            best_score = -1
            for _ in range(revise_k):
                cand_text = _sample_line_action(
                    model, tokenizer, FILER_LINE_SYSTEM_PROMPT, r_user,
                    args.filer_max_new_tokens, args, device,
                    temperature=revise_temp,
                )
                cand_line = try_parse_line_draft(cand_text, lid)
                if cand_line is None:
                    score = 0
                else:
                    cand_tmp = DraftReturn(return_version="ty24-v1", lines=[cand_line], metadata={})
                    lands = bool(line_strict_correctness_by_line_id(cand_tmp, xml_path).get(lid, False))
                    score = 2 if lands else 1
                if score > best_score:
                    best_score = score
                    best_text = cand_text
                    best_line = cand_line
                    if score == 2:
                        break
            revise_text = best_text
            z_line = best_line
            revise_parsed = z_line is not None
            if z_line is not None:
                S_z = 1.0 if best_score == 2 else 0.0

            cf_keep, cf_revise = (0.0, 1.0) if sac_correct_line else (1.0, 0.0)
            keep_reward = cf_keep + outcome_bonus * float(S_x)
            if z_line is not None:
                revise_reward = cf_revise + outcome_bonus * float(S_z)
            else:
                revise_reward = cf_revise - fmt_pen

            _set_active_adapter(model, _ROLES["approver_adapter"])
            a_user = _line_approver_user_payload(
                input_json, x_line, lid, y_raw, z_line, context_so_far[:-1],
            )
            approver_text = _sample_line_action(
                model, tokenizer, APPROVER_LINE_SYSTEM_PROMPT, a_user, 256, args, device,
            )
            parsed_decision = _parse_approver_decision(approver_text)
            approver_parsed = parsed_decision is not None
            revision_decision = parsed_decision if parsed_decision is not None else "KEEP"

            # epsilon-exploration: with prob eps force REVISE so case-5a/5b
            # transitions surface even after the approver collapses to KEEP.
            # The paired-action GRPO loss is computed against canonical KEEP
            # / REVISE strings so this never produces an off-policy gradient.
            if explore_eps > 0.0 and revision_decision != "REVISE":
                if random.random() < explore_eps:
                    revision_decision = "REVISE"
                    forced_explore = True

            # Counterfactual approver reward: the right meta-action depends
            # on whether the SAC's *suggestion* is good (S_z), not just on
            # whether SAC was justified. KEEP and REVISE both get a learned
            # action stub every transition.
            if z_line is None:
                approver_keep_reward = 1.0
                approver_revise_reward = 0.0 - fmt_pen
            elif not sac_correct_line:
                # Filer correct, V SAC: false-positive SAC. KEEP is right.
                approver_keep_reward = 1.0 + outcome_bonus * float(S_x)
                if S_z < 0.5:
                    approver_revise_reward = -cave_penalty
                else:
                    approver_revise_reward = 0.0 + outcome_bonus * float(S_z)
            else:
                # Filer wrong: SAC justified; suggestion quality picks 5a/5b/6a/6b.
                if S_z > 0.5:
                    approver_keep_reward = 0.0 - miss_penalty
                    approver_revise_reward = 1.0 + outcome_bonus * float(S_z)
                else:
                    approver_keep_reward = 1.0
                    approver_revise_reward = 0.0 + outcome_bonus * float(S_z)

        # post_revise draft is for measurement only; never threaded back into context_so_far.
        if revision_considered and revision_decision == "REVISE" and z_line is not None:
            post_revise_lines.append(z_line)
        else:
            post_revise_lines.append(x_line)

        # 8-bucket linewise taxonomy {1, 2, 3, 4, 5a, 5b, 6a, 6b}, mutually exclusive.
        rx_b = bool(S_x > 0.5)
        rz_b = bool(S_z > 0.5)
        case_id = 0
        case_sub = "other"
        decision_was_right: Optional[bool] = None
        if not revision_considered:
            case_id = 1 if rx_b else 4
        elif rx_b:
            case_id = 2 if revision_decision == "REVISE" else 3
            decision_was_right = revision_decision == "KEEP"
        else:
            if revision_decision == "REVISE":
                case_id = 5
                case_sub = "5a" if rz_b else "5b"
                decision_was_right = rz_b
            else:
                case_id = 6
                case_sub = "6a" if not rz_b else "6b"
                decision_was_right = not rz_b

        transitions.append({
            "iter_id": iter_id,
            "s_id": str(xml_path.parent.resolve()),
            "line_id": lid,
            "filer_line_raw": f_text,
            "filer_parsed": bool(filer_parsed),
            "filer_line_struct": _line_dict(x_line),
            "y_choice": y_choice,
            "y_text": y_text,
            "oracle": {
                "S_x_line": S_x,
                "S_z_line": S_z,
                "sac_correct_line": bool(sac_correct_line),
            },
            "verifier_rewards": verifier_pair_rewards,
            "revision": {
                "considered": revision_considered,
                "parsed": revise_parsed,
                "keep_text": json.dumps(_line_dict(x_line), ensure_ascii=False),
                "revise_text": revise_text,
                "keep_reward": keep_reward,
                "revise_reward": revise_reward,
                "decision": revision_decision,
            },
            "approver": {
                "considered": revision_considered,
                "parsed": approver_parsed,
                "text": approver_text,
                "decision": revision_decision,
                "keep_reward": approver_keep_reward,
                "revise_reward": approver_revise_reward,
                "sac_was_correct": bool(sac_correct_line),
                "decision_was_right": decision_was_right,
                "forced_explore": forced_explore,
            },
            "prompts": {
                "filer_user": f_user,
                "verifier_user": v_user,
                "revise_user": r_user,
                "approver_user": a_user,
            },
            "case_sub": case_sub,
            "case_id": int(case_id),
        })

    draft_x = DraftReturn(return_version="ty24-v1", lines=selected_lines, metadata={})
    draft_post = DraftReturn(return_version="ty24-v1", lines=post_revise_lines, metadata={})
    return transitions, draft_x, draft_post


# -----------------------------------------------------------------------------
# GRPO update over a batch of paired-action transitions
# -----------------------------------------------------------------------------

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
    """For each transition, run up to three paired-action GRPO updates:
    verifier {SAC,NS}, filer-revise {KEEP,REVISE}, approver {KEEP,REVISE}.
    """
    if not transitions:
        return 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    lf_total = 0.0
    lv_total = 0.0
    n_backward = 0

    skip_v_on_invalid = bool(args.skip_verifier_update_on_invalid_filer)
    skip_f_on_invalid_revise = bool(args.skip_filer_revise_update_on_invalid_revise)

    # Approver homeostasis: when the step's REVISE rate is below the target,
    # add a uniform bonus to all revise rewards. Inverse-frequency loss
    # weighting amplifies the gradient on rare cases (5a in particular).
    target_rev_rate = float(args.approver_target_revise_rate)
    balance_strength = float(args.approver_balance_strength)
    rare_alpha = float(args.approver_rare_case_weight)
    considered = [t for t in transitions if (t.get("approver") or {}).get("considered", False)]
    n_considered = len(considered)
    if n_considered > 0 and balance_strength > 0:
        n_revise = sum(
            1 for t in considered
            if str((t.get("approver") or {}).get("decision", "KEEP")).upper() == "REVISE"
        )
        revise_share = n_revise / float(n_considered)
        revise_bonus = balance_strength * max(target_rev_rate - revise_share, 0.0)
    else:
        revise_bonus = 0.0
    if n_considered > 0 and rare_alpha > 0:
        case_counts: Dict[int, int] = {}
        for t in considered:
            cid = int(t.get("case_id", 0))
            case_counts[cid] = case_counts.get(cid, 0) + 1
        avg_count = n_considered / float(max(len(case_counts), 1))
    else:
        case_counts = {}
        avg_count = 1.0

    for tr in transitions:
        lid = tr["line_id"]
        filer_parsed = bool(tr.get("filer_parsed", True))
        revise_parsed = tr.get("revision", {}).get("parsed", None)

        # Verifier paired-action update: SAC vs NO_SAC against the oracle label.
        # Skip on invalid-filer-line rows (those carry an artificial sac=True
        # label and would otherwise reward always-SAC).
        if not (skip_v_on_invalid and not filer_parsed):
            _set_active_adapter(model, _ROLES["verifier_adapter"])
            v_user = tr["prompts"]["verifier_user"]
            v_actions = [
                json.dumps({"target_line": lid, "decision": "SAC"}, ensure_ascii=False),
                json.dumps({"target_line": lid, "decision": "NO_SAC"}, ensure_ascii=False),
            ]
            seqs_v: List[torch.Tensor] = []
            pl_v: List[int] = []
            for act in v_actions:
                seq, pl = _build_seq_from_prompt_and_action(
                    tokenizer, VERIFIER_LINE_SYSTEM_PROMPT, v_user, act, device,
                )
                seqs_v.append(seq)
                pl_v.append(pl)
            rvs = torch.tensor(
                [tr["verifier_rewards"]["SAC"], tr["verifier_rewards"]["NO_SAC"]],
                device=device, dtype=torch.float32,
            )
            adv_v = (rvs - rvs.mean()) / (rvs.std(unbiased=False) + 1e-8)
            lv_total += grpo_group_accumulate_gradients(
                model, seqs_v, pl_v, adv_v, args.beta_kl, device, use_amp, scaler,
                loss_scale=float(args.verifier_loss_scale),
            )
            n_backward += 1

        # Filer KEEP/REVISE paired update on the revise prompt.
        do_filer_revise = bool(tr["revision"]["considered"]) and not (
            skip_f_on_invalid_revise and revise_parsed is False
        )
        if do_filer_revise:
            _set_active_adapter(model, _ROLES["filer_adapter"])
            f_user = tr["prompts"]["revise_user"]
            keep_text = tr["revision"]["keep_text"]
            revise_text = tr["revision"]["revise_text"] or keep_text
            seq_keep, pl_keep = _build_seq_from_prompt_and_action(
                tokenizer, FILER_LINE_SYSTEM_PROMPT, f_user, keep_text, device,
            )
            seq_rev, pl_rev = _build_seq_from_prompt_and_action(
                tokenizer, FILER_LINE_SYSTEM_PROMPT, f_user, revise_text, device,
            )
            rfs = torch.tensor(
                [tr["revision"]["keep_reward"], tr["revision"]["revise_reward"]],
                device=device, dtype=torch.float32,
            )
            adv_f = (rfs - rfs.mean()) / (rfs.std(unbiased=False) + 1e-8)
            lf_total += grpo_group_accumulate_gradients(
                model, [seq_keep, seq_rev], [pl_keep, pl_rev],
                adv_f, args.beta_kl, device, use_amp, scaler,
            )
            n_backward += 1

        # Approver KEEP/REVISE paired update (always trains both stubs counterfactually).
        ap = tr.get("approver", {}) or {}
        if bool(ap.get("considered", False)):
            a_user = tr["prompts"].get("approver_user", "") or ""
            if a_user:
                _set_active_adapter(model, _ROLES["approver_adapter"])
                ap_actions = [
                    json.dumps({"target_line": lid, "decision": "KEEP"}, ensure_ascii=False),
                    json.dumps({"target_line": lid, "decision": "REVISE"}, ensure_ascii=False),
                ]
                seqs_a: List[torch.Tensor] = []
                pl_a: List[int] = []
                for act in ap_actions:
                    seq, pl = _build_seq_from_prompt_and_action(
                        tokenizer, APPROVER_LINE_SYSTEM_PROMPT, a_user, act, device,
                    )
                    seqs_a.append(seq)
                    pl_a.append(pl)
                ap_keep_r = float(ap.get("keep_reward", 0.0))
                ap_rev_r = float(ap.get("revise_reward", 0.0)) + revise_bonus
                ras = torch.tensor([ap_keep_r, ap_rev_r], device=device, dtype=torch.float32)
                adv_a = (ras - ras.mean()) / (ras.std(unbiased=False) + 1e-8)
                if rare_alpha > 0 and case_counts:
                    cid = int(tr.get("case_id", 0))
                    cnt = float(case_counts.get(cid, 1))
                    rarity_scale = (avg_count / max(cnt, 1.0)) ** rare_alpha
                else:
                    rarity_scale = 1.0
                lf_total += grpo_group_accumulate_gradients(
                    model, seqs_a, pl_a, adv_a, args.beta_kl, device, use_amp, scaler,
                    loss_scale=float(rarity_scale),
                )
                n_backward += 1

    # Empty step: e.g. a case where every line had invalid filer JSON AND no
    # SAC raises. Calling scaler.step() then asserts; skip cleanly.
    if n_backward == 0:
        return 0.0, 0.0
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return lf_total, lv_total


# -----------------------------------------------------------------------------
# Periodic eval on the held-out test split
# -----------------------------------------------------------------------------

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
    if not test_cases:
        return None
    max_cases = int(args.eval_max_cases)
    selected = test_cases[: max_cases if max_cases > 0 else len(test_cases)]
    if not selected:
        return None
    prev_train = model.training
    prev_temp, prev_top_p = float(args.temperature), float(args.top_p)
    args.temperature = float(args.eval_temperature)
    args.top_p = float(args.eval_top_p)
    model.eval()
    recs: List[Dict[str, Any]] = []
    for case_dir in selected:
        input_json = load_case_input(case_dir)
        xml_path = case_dir / "output.xml"
        transitions, draft_x, draft_post = _dual_linewise_rollout_transitions(
            model, tokenizer, input_json, xml_path, args, device, gstep,
        )
        case_hist = {"1": 0, "2": 0, "3": 0, "4": 0, "5a": 0, "5b": 0, "6a": 0, "6b": 0}
        for tr in transitions:
            cid = int(tr.get("case_id", 0))
            csub = str(tr.get("case_sub", "other"))
            if cid in (1, 2, 3, 4):
                case_hist[str(cid)] += 1
            elif cid in (5, 6) and csub in ("5a", "5b", "6a", "6b"):
                case_hist[csub] += 1
        acc = float(reward_filer_draft(draft_x, xml_path))
        acc_post = float(reward_filer_draft(draft_post, xml_path))
        lc = line_strict_correctness_by_line_id(draft_x, xml_path)
        lc_post = line_strict_correctness_by_line_id(draft_post, xml_path)
        line_acc = sum(1.0 for ok in lc.values() if ok) / max(len(lc), 1) if lc else 0.0
        line_acc_post = sum(1.0 for ok in lc_post.values() if ok) / max(len(lc_post), 1) if lc_post else 0.0
        recs.append({
            "case_dir": str(case_dir.resolve()),
            "final_draft_accuracy_strict": acc,
            "final_draft_accuracy_post_revise": acc_post,
            "final_draft_line_accuracy": float(line_acc),
            "final_draft_line_accuracy_post_revise": float(line_acc_post),
            "transitions_evaluated": len(transitions),
            "case_histogram": case_hist,
        })
    agg = {k: 0 for k in ("1", "2", "3", "4", "5a", "5b", "6a", "6b")}
    for r in recs:
        for k in agg:
            agg[k] += int(r["case_histogram"].get(k, 0))
    n = max(len(recs), 1)
    out = {
        "step": int(gstep),
        "utc": datetime.now(timezone.utc).isoformat(),
        "n_test_cases_evaluated": len(recs),
        "test_accuracy_strict_mean": sum(r["final_draft_accuracy_strict"] for r in recs) / n,
        "test_accuracy_strict_post_revise_mean": sum(r["final_draft_accuracy_post_revise"] for r in recs) / n,
        "test_line_accuracy_mean": sum(r["final_draft_line_accuracy"] for r in recs) / n,
        "test_line_accuracy_post_revise_mean": sum(r["final_draft_line_accuracy_post_revise"] for r in recs) / n,
        "test_case_histogram": agg,
        "per_case": recs,
    }
    eval_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
    eval_fp.flush()
    args.temperature = prev_temp
    args.top_p = prev_top_p
    if prev_train:
        model.train()
    return out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def _build_optimizer_with_all_adapter_params(
    model: torch.nn.Module, lr: float, weight_decay: float,
) -> Tuple[AdamW, List[torch.nn.Parameter]]:
    """Collect LoRA params across BOTH adapters into the optimizer.

    PEFT toggles requires_grad per active adapter; a naive scan would only
    capture the currently active adapter and silently drop the other one.
    """
    trainable: List[torch.nn.Parameter] = []
    seen: set = set()
    prev_active = getattr(model, "active_adapter", "default")
    for name in ("default", "theta"):
        try:
            model.set_adapter(name)
        except Exception:
            continue
        for p in model.parameters():
            if p.requires_grad and id(p) not in seen:
                trainable.append(p)
                seen.add(id(p))
    try:
        model.set_adapter(prev_active)
    except Exception:
        model.set_adapter("default")
    for p in trainable:
        p.requires_grad = True
    return AdamW(trainable, lr=lr, weight_decay=weight_decay), trainable


def _setup_model(args: argparse.Namespace, device: str) -> Tuple[Any, Any]:
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model_path = _resolve_model_path(args.model_path)
    print(f"[dpa-grpo] device={device} model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.train()

    resume_from = (args.resume_from or "").strip()
    adapter_resume: Optional[Path] = None
    if resume_from:
        adapter_resume = Path(resume_from).resolve() / "adapter_final"
        if not adapter_resume.is_dir():
            raise SystemExit(f"--resume-from: expected adapter at {adapter_resume}")
        print(f"[dpa-grpo] resuming LoRA from {adapter_resume}")
        model = PeftModel.from_pretrained(model, str(adapter_resume), is_trainable=True)
        model.print_trainable_parameters()
    else:
        targets = _lora_target_modules(model)
        if not targets:
            raise RuntimeError("Could not infer LoRA target modules for this model.")
        print(f"[dpa-grpo] LoRA targets: {targets}")
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
        )
        model = get_peft_model(model, lcfg)
        model.add_adapter("theta", lcfg)
        model.set_adapter("default")
        print("[dpa-grpo] dual adapters: filer/approver=default, verifier=theta")
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[dpa-grpo] gradient checkpointing on")
        except Exception as e:
            print(f"[dpa-grpo] gradient_checkpointing_enable failed: {e}")
    return model, tokenizer


def train_loop(args: argparse.Namespace) -> None:
    device = resolve_torch_device()
    model, tokenizer = _setup_model(args, device)
    optimizer, trainable = _build_optimizer_with_all_adapter_params(
        model, args.lr, args.weight_decay,
    )
    print(
        f"[dpa-grpo] optimizer params: {len(trainable)} tensors, "
        f"{sum(p.numel() for p in trainable):,} elements"
    )

    # bf16 doesn't need a GradScaler (the unscale kernel is fp16-only); we
    # still autocast to bf16 below if --amp is set.
    scaler: Optional[Any] = None

    global_step_base = 0
    resume_from = (args.resume_from or "").strip()
    if resume_from:
        state_pt = Path(resume_from).resolve() / "training_state.pt"
        if state_pt.is_file():
            sd = torch.load(state_pt, map_location=device)
            try:
                optimizer.load_state_dict(sd["optimizer"])
            except (ValueError, RuntimeError) as e:
                print(f"[dpa-grpo] could not load optimizer state ({e}); using fresh AdamW")
            global_step_base = int(sd.get("last_step", 0))
            print(f"[dpa-grpo] resumed at step={global_step_base}")

    cases_all = discover_case_dirs(Path(args.cases_root))
    if not cases_all:
        raise SystemExit(f"No cases under {args.cases_root}")
    train_cases, test_cases = split_train_test_cases(
        cases_all, args.test_fraction, args.split_seed,
    )
    if not train_cases:
        raise SystemExit("Train split is empty.")
    print(
        f"[dpa-grpo] cases: {len(cases_all)} total -> train={len(train_cases)} "
        f"test={len(test_cases)} (test_fraction={args.test_fraction}, seed={args.split_seed})"
    )

    full_train_cases = list(train_cases)
    subset_size = int(args.train_case_subset_size or 0)
    resample_each_step = bool(args.train_case_resample_each_step)
    if subset_size > 0 and not resample_each_step:
        train_cases = list(train_cases)[:subset_size]
        print(f"[dpa-grpo] train subset: {len(train_cases)} of {len(full_train_cases)} (cycle={args.train_case_cycle})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (args.run_name or "").strip() or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = run_dir / "step_traces"

    (run_dir / "run_config.json").write_text(
        json.dumps({
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": _resolve_model_path(args.model_path),
            "cases_root": str(Path(args.cases_root).resolve()),
            "n_train_cases_full": len(full_train_cases),
            "n_train_cases": len(train_cases),
            "n_test_cases": len(test_cases),
            "train_case_dirs": [str(p.resolve()) for p in train_cases],
            "test_case_dirs": [str(p.resolve()) for p in test_cases],
            "global_step_base": global_step_base,
            "args": vars(args),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    metrics_path = run_dir / "metrics.jsonl"
    eval_path = run_dir / "eval_metrics.jsonl"
    tsv_path = run_dir / "rounds.tsv"
    print(f"[dpa-grpo] metrics={metrics_path}\n[dpa-grpo] eval={eval_path}")

    rng = random.Random(args.seed)
    use_amp = scaler is not None and device.startswith("cuda")

    metrics_fp = open(metrics_path, "w", encoding="utf-8")
    eval_fp = open(eval_path, "w", encoding="utf-8")
    tsv_fp = open(tsv_path, "w", encoding="utf-8")
    tsv_fp.write("global_step\tcase\tloss\tfiler_r\tver_r\tacc\tacc_post\n")
    try:
        _training_steps_dual(
            args, model, tokenizer, optimizer, scaler,
            train_cases, full_train_cases, rng, device, use_amp,
            run_dir, trace_dir, metrics_fp, tsv_fp, eval_fp,
            global_step_base, test_cases,
        )
    finally:
        metrics_fp.close()
        eval_fp.close()
        tsv_fp.close()
        print(f"[dpa-grpo] artifacts at {run_dir.resolve()}")


def _training_steps_dual(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: AdamW,
    scaler: Optional[Any],
    cases: List[Path],
    full_train_cases: List[Path],
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
    train_all = bool(args.train_all_cases_per_step)
    resample_each_step = bool(args.train_case_resample_each_step)
    subset_size = int(args.train_case_subset_size or 0)

    eval_state: Optional[Dict[str, Any]] = None

    for step in range(1, args.steps + 1):
        gstep = global_step_base + step

        if resample_each_step and subset_size > 0:
            k = min(subset_size, len(full_train_cases))
            step_cases = rng.sample(list(full_train_cases), k)
        elif train_all:
            step_cases = list(cases)
        else:
            step_cases = [_select_case_for_step(cases, step, bool(args.train_case_cycle), rng)]

        transitions: List[Dict[str, Any]] = []
        lf = 0.0
        lv = 0.0
        per_case_records: List[Dict[str, Any]] = []
        primary_case_dir: Path = step_cases[-1]

        for sub_idx, case_dir in enumerate(step_cases):
            input_json = load_case_input(case_dir)
            xml_path = case_dir / "output.xml"
            sub_transitions, draft_x, draft_post = _dual_linewise_rollout_transitions(
                model, tokenizer, input_json, xml_path, args, device, gstep,
            )
            slf, slv = _apply_transition_batch_update(
                model, tokenizer, optimizer, scaler,
                sub_transitions, args, device, use_amp,
            )
            sub_acc = float(reward_filer_draft(draft_x, xml_path))
            sub_acc_post = float(reward_filer_draft(draft_post, xml_path))
            sub_lc = line_strict_correctness_by_line_id(draft_x, xml_path)
            sub_lc_post = line_strict_correctness_by_line_id(draft_post, xml_path)
            per_case_records.append({
                "sub_idx": sub_idx,
                "case_dir": str(case_dir.resolve()),
                "n_transitions": len(sub_transitions),
                "loss_filer_term": float(slf),
                "loss_verifier_term": float(slv),
                "final_draft_accuracy_strict": sub_acc,
                "final_draft_accuracy_post_revise": sub_acc_post,
                "final_draft_line_correctness": sub_lc,
                "final_draft_line_correctness_post_revise": sub_lc_post,
            })
            transitions.extend(sub_transitions)
            lf += slf
            lv += slv
            primary_case_dir = case_dir

        # Step-level diagnostics: 8-bucket case histogram + reward / decision rates.
        case_hist = {"1": 0, "2": 0, "3": 0, "4": 0, "5a": 0, "5b": 0, "6a": 0, "6b": 0}
        filer_rewards: List[float] = []
        verifier_rewards: List[float] = []
        approver_rewards: List[float] = []
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
            verifier_rewards.append(float(vr.get(y_choice, 0.0)))
            rev = tr.get("revision", {}) or {}
            ap = tr.get("approver", {}) or {}
            if bool(rev.get("considered", False)) and str(rev.get("decision", "KEEP")) == "REVISE":
                filer_rewards.append(float(rev.get("revise_reward", 0.0)))
            else:
                filer_rewards.append(float(rev.get("keep_reward", 0.0)))
            if bool(ap.get("considered", False)):
                if str(ap.get("decision", "KEEP")) == "REVISE":
                    approver_rewards.append(float(ap.get("revise_reward", 0.0)))
                else:
                    approver_rewards.append(float(ap.get("keep_reward", 0.0)))
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
        dec_total = float(max(n_decision_right + n_decision_wrong, 1))

        sac_rate = n_sac / denom
        invalid_filer_rate = n_filer_invalid / denom
        invalid_revise_rate = n_revise_invalid / rev_denom
        revise_rate = n_revise_chosen / rev_denom
        invalid_approver_rate = n_approver_invalid / rev_denom
        decision_accuracy = n_decision_right / dec_total
        forced_revise_rate = n_forced_explore / rev_denom

        final_acc = sum(pc["final_draft_accuracy_strict"] for pc in per_case_records) / max(len(per_case_records), 1)
        final_acc_post = sum(pc["final_draft_accuracy_post_revise"] for pc in per_case_records) / max(len(per_case_records), 1)
        final_lc = per_case_records[-1]["final_draft_line_correctness"] if per_case_records else {}
        final_lc_post = per_case_records[-1].get("final_draft_line_correctness_post_revise", {}) if per_case_records else {}

        rec = {
            "step": gstep,
            "local_step": step,
            "utc": datetime.now(timezone.utc).isoformat(),
            "case_dir": str(primary_case_dir.resolve()),
            "case_dirs": [pc["case_dir"] for pc in per_case_records],
            "n_cases_in_step": len(per_case_records),
            "loss_total": float(lf + lv),
            "loss_filer_term": float(lf),
            "loss_verifier_term": float(lv),
            "filer_reward_mean": sum(filer_rewards) / max(len(filer_rewards), 1),
            "verifier_reward_mean": sum(verifier_rewards) / max(len(verifier_rewards), 1),
            "approver_reward_mean": sum(approver_rewards) / max(len(approver_rewards), 1) if approver_rewards else 0.0,
            "filer_rewards": filer_rewards,
            "verifier_rewards": verifier_rewards,
            "approver_rewards": approver_rewards,
            "final_draft_accuracy_strict": float(final_acc),
            "final_draft_accuracy_per_case": [pc["final_draft_accuracy_strict"] for pc in per_case_records],
            "final_draft_accuracy_post_revise": float(final_acc_post),
            "final_draft_accuracy_post_revise_per_case": [pc["final_draft_accuracy_post_revise"] for pc in per_case_records],
            "transitions_added": len(transitions),
            "case_histogram": case_hist,
            "sac_rate": sac_rate,
            "invalid_filer_json_rate": invalid_filer_rate,
            "revision_considered_count": int(n_revise_considered),
            "invalid_revise_json_rate": invalid_revise_rate,
            "revise_rate": revise_rate,
            "invalid_approver_json_rate": invalid_approver_rate,
            "decision_accuracy": decision_accuracy,
            "decision_right_count": int(n_decision_right),
            "decision_wrong_count": int(n_decision_wrong),
            "approver_forced_revise_rate": forced_revise_rate,
            "approver_forced_revise_count": int(n_forced_explore),
        }
        metrics_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        metrics_fp.flush()

        step_label = primary_case_dir.name if len(per_case_records) <= 1 else f"{len(per_case_records)}cases"
        tsv_fp.write(
            f"{gstep}\t{step_label}\t{float(lf+lv):.6f}\t"
            f"{rec['filer_reward_mean']:.4f}\t{rec['verifier_reward_mean']:.4f}\t"
            f"{final_acc:.4f}\t{final_acc_post:.4f}\n"
        )
        tsv_fp.flush()

        _write_step_trace(trace_dir, gstep, {
            "step": gstep,
            "utc": rec["utc"],
            "case_dir": str(primary_case_dir.resolve()),
            "case_dirs": [pc["case_dir"] for pc in per_case_records],
            "per_case": per_case_records,
            "transitions": transitions,
            "case_histogram": case_hist,
            "sac_rate": sac_rate,
            "revise_rate": revise_rate,
            "decision_accuracy": decision_accuracy,
            "final_draft_accuracy_strict": float(final_acc),
            "final_draft_accuracy_post_revise": float(final_acc_post),
            "final_draft_line_correctness": final_lc,
            "final_draft_line_correctness_post_revise": final_lc_post,
        })

        if step % args.log_every == 0:
            forced = f" forced_rev={forced_revise_rate:.2f}" if float(args.approver_explore_eps) > 0.0 else ""
            print(
                f"[step {gstep} ({step}/{args.steps})] cases={len(per_case_records)} "
                f"loss={float(lf+lv):.4f} sac={sac_rate:.2f} rev_rate={revise_rate:.2f} "
                f"dec_acc={decision_accuracy:.2f}{forced} "
                f"acc={final_acc:.3f} acc_post={final_acc_post:.3f}"
            )

        if step % args.save_every == 0:
            save_path = run_dir / f"adapter_step_{gstep}"
            _save_lora_checkpoint(model, tokenizer, run_dir, save_path, gstep, optimizer, scaler)
            print(f"[dpa-grpo] saved {save_path}")

        if int(args.eval_every) > 0 and test_cases and step % int(args.eval_every) == 0:
            eval_out = _run_dual_periodic_eval(
                model, tokenizer, test_cases, args, device, gstep, eval_fp,
            )
            if eval_out is not None:
                if eval_state is None:
                    metric_key = (
                        "test_accuracy_strict_post_revise_mean"
                        if str(args.best_metric) == "post_revise"
                        else "test_accuracy_strict_mean"
                    )
                    eval_state = {
                        "best_score": float("-inf"),
                        "best_step": 0,
                        "evals_since_improvement": 0,
                        "metric_key": metric_key,
                    }
                metric_key = eval_state["metric_key"]
                score = float(eval_out.get(metric_key, 0.0))
                if score > eval_state["best_score"] + 1e-12:
                    eval_state["best_score"] = score
                    eval_state["best_step"] = gstep
                    eval_state["evals_since_improvement"] = 0
                    if bool(args.save_best_adapter):
                        best_path = run_dir / f"best_adapter_step_{gstep}"
                        _save_lora_checkpoint(
                            model, tokenizer, run_dir, best_path, gstep, optimizer, scaler,
                        )
                        (run_dir / "best.json").write_text(
                            json.dumps({
                                "best_step": gstep,
                                "best_score": score,
                                "best_metric": metric_key,
                                "best_path": str(best_path),
                            }, indent=2),
                            encoding="utf-8",
                        )
                        print(f"[dpa-grpo] NEW BEST {metric_key}={score:.4f} at step {gstep}")
                else:
                    eval_state["evals_since_improvement"] += 1
                    print(
                        f"[dpa-grpo] eval step {gstep}: {metric_key}={score:.4f} "
                        f"(best={eval_state['best_score']:.4f} at {eval_state['best_step']}; "
                        f"stale={eval_state['evals_since_improvement']})"
                    )
                patience = int(args.early_stop_patience)
                if (
                    patience > 0
                    and gstep >= int(args.early_stop_min_step)
                    and eval_state["evals_since_improvement"] >= patience
                ):
                    print(
                        f"[dpa-grpo] EARLY STOP at step {gstep}: no improvement for "
                        f"{eval_state['evals_since_improvement']} evals (patience={patience}); "
                        f"best {metric_key}={eval_state['best_score']:.4f} at step {eval_state['best_step']}"
                    )
                    (run_dir / "early_stopped.json").write_text(
                        json.dumps({
                            "stopped_at_step": gstep,
                            "best_step": eval_state["best_step"],
                            "best_score": eval_state["best_score"],
                            "best_metric": metric_key,
                            "patience": patience,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    break

    final = run_dir / "adapter_final"
    g_final = global_step_base + args.steps
    _save_lora_checkpoint(model, tokenizer, run_dir, final, g_final, optimizer, scaler)
    print(f"[dpa-grpo] saved final adapter at step {g_final}: {final}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPA-GRPO training")

    p.add_argument(
        "--cases-root", type=str,
        default=str(_root / "tax-calc-bench" / "tax_calc_bench" / "ty24" / "test_data"),
        help="Root containing per-case dirs (input.json + output.xml)",
    )
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-case-subset-size", type=int, default=0)
    p.add_argument("--train-case-cycle", action="store_true", default=False)
    p.add_argument("--train-all-cases-per-step", action="store_true", default=False)
    p.add_argument(
        "--train-case-resample-each-step", action="store_true", default=False,
        help="Per-step random sub-sampling from the full train pool (avoids "
        "memorising a fixed subset).",
    )

    p.add_argument(
        "--model-path", type=str,
        default=os.environ.get("TAX_MODEL_PATH", "Qwen/Qwen3-8B"),
        help="Either a HuggingFace model id (e.g. 'Qwen/Qwen3-8B') or a "
        "local snapshot directory.",
    )
    p.add_argument("--output-dir", type=str, default="grpo_checkpoints")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--steps", type=int, default=30)

    p.add_argument("--group-size", type=int, default=2,
                   help="K rollouts per GRPO group; paired-action GRPO uses 2.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta-kl", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--filer-max-new-tokens", type=int, default=2048)
    p.add_argument("--verifier-max-new-tokens", type=int, default=2048)
    p.add_argument("--invalid-json-reward", type=float, default=-0.5)
    p.add_argument("--revise-format-penalty", type=float, default=0.25)

    p.add_argument("--skip-verifier-update-on-invalid-filer", action="store_true", default=True)
    p.add_argument("--skip-filer-revise-update-on-invalid-revise", action="store_true", default=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument(
        "--resume-from", type=str, default="",
        help="Previous run dir containing adapter_final/; loads LoRA + optional training_state.pt",
    )

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--amp", action="store_true", default=False, help="bf16 autocast on CUDA")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)

    # GRPO loss-weight knobs (case taxonomy is always-on for paired-action mode).
    p.add_argument("--filer-outcome-bonus", type=float, default=0.5)
    p.add_argument("--verifier-loss-scale", type=float, default=0.5)
    p.add_argument("--revise-sample-k", type=int, default=5)
    p.add_argument("--revise-temperature", type=float, default=0.7)

    # Approver shaping (DPA-GRPO design choices).
    p.add_argument("--approver-cave-penalty", type=float, default=0.2)
    p.add_argument("--approver-miss-penalty", type=float, default=0.3)
    p.add_argument("--approver-target-revise-rate", type=float, default=0.30)
    p.add_argument("--approver-balance-strength", type=float, default=1.5)
    p.add_argument("--approver-rare-case-weight", type=float, default=1.0)
    p.add_argument(
        "--approver-explore-eps", type=float, default=0.4,
        help="Forced REVISE probability during training rollouts; surfaces case-5 transitions "
        "even after the approver collapses to KEEP. The paired-action loss is unchanged.",
    )

    # Eval / best-checkpoint tracking.
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-max-cases", type=int, default=8)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)
    p.add_argument("--save-best-adapter", action="store_true", default=True)
    p.add_argument("--no-save-best-adapter", dest="save_best_adapter", action="store_false")
    p.add_argument(
        "--best-metric", type=str, default="post_revise",
        choices=("post_revise", "strict"),
        help="Metric tracked for best-checkpoint and early-stop.",
    )
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-min-step", type=int, default=10)

    # Episode/per-line case taxonomy thresholds (used by the eval-side
    # classify_episode helper exposed for downstream analysis; defaults from
    # CaseTaxonomyConfig). Kept as flags for parity with paper Table 4.
    p.add_argument("--case-x-good", type=float, default=CaseTaxonomyConfig.x_good)
    p.add_argument("--case-verifier-good", type=float, default=CaseTaxonomyConfig.verifier_good)
    p.add_argument("--case-yprime-margin", type=float, default=CaseTaxonomyConfig.y_prime_gt_y_margin)
    p.add_argument("--case-revise-draft-amount-abs-tol", type=float, default=1e-2)
    p.add_argument("--line-case-z-improve-eps", type=float, default=1e-4)
    p.add_argument("--line-case-amount-tol", type=float, default=1e-6)
    p.add_argument("--line-case-reward-scale", type=float, default=0.15)
    p.add_argument("--line-case-prefer-boost", type=float, default=0.15)

    # Aliases retained for shell-script compatibility (no-ops at the
    # algorithmic level: linewise rollout and case taxonomy are mandatory).
    p.add_argument("--linewise-rollout", action="store_true", default=True)
    p.add_argument("--case-taxonomy", action="store_true", default=True)
    p.add_argument("--line-case-taxonomy", action="store_true", default=True)
    p.add_argument("--alternate-rounds", action="store_true", default=False,
                   help="Retained for compatibility; paired-action GRPO doesn't need it.")

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 for GRPO normalization")
    train_loop(args)


if __name__ == "__main__":
    main()
