"""DPA-GRPO: training module."""
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

from core.client import extract_json_from_response
from core.device_pick import resolve_torch_device
from core.rewards import reward_filer_draft
from benchmarks.line_eval import line_strict_correctness_by_line_id
from benchmarks.registry import get_dataset_config
from benchmarks.tax.tax_case_taxonomy import (
    CaseTaxonomyConfig,
    classify_episode,
    default_case_loss_weights,
    dual_verifier_rewards_from_raw,
)
from benchmarks.tax.tax_line_case import (
    LineCaseConfig,
    case_preference_score,
    line_case_loss_weights,
    per_line_case_counts_from_verifier_raw,
)
from dpa_grpo.checkpointing import (
    _HAS_PEFT, LoraConfig, PeftModel, _lora_target_modules, _resolve_model_path,
    _save_lora_checkpoint, _save_training_state, _set_active_adapter, get_peft_model,
)
from dpa_grpo.config import configure_dataset, _ds_cfg, _gold_path
from dpa_grpo.data import (
    _draft_to_dict,
    _select_case_for_step,
    _write_step_trace,
    discover_case_dirs,
    load_case_input,
    split_train_test_cases,
)
from dpa_grpo.dual_rollout import _dual_linewise_rollout_transitions
from dpa_grpo.eval import _run_dual_periodic_eval
from dpa_grpo.losses import grpo_group_accumulate_gradients
from dpa_grpo.paths import checkpoint_root
from dpa_grpo.rollouts import (
    filer_grpo_rollouts,
    linewise_filer_verifier_rollouts,
    revise_filer_sample_for_taxonomy,
    verifier_grpo_rollouts,
)
from dpa_grpo.sampling import _role_assignment
from dpa_grpo.updates import _apply_transition_batch_update


def _train_best_metric_key(args: argparse.Namespace) -> str:
    if str(getattr(args, "best_metric", "post_revise")) == "post_revise":
        return "train_accuracy_post_revise"
    return "train_accuracy_strict"


def _maybe_save_best_train_checkpoint(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    run_dir: Path,
    gstep: int,
    score: float,
    train_best_state: Dict[str, Any],
    optimizer: AdamW,
    scaler: Optional[Any],
) -> None:
    """Save adapter when train-batch accuracy improves (used when eval is off)."""
    if not args.use_lora or not bool(getattr(args, "save_best_adapter", True)):
        return
    metric_key = train_best_state["metric_key"]
    if score <= float(train_best_state["best_score"]) + 1e-12:
        return
    train_best_state["best_score"] = float(score)
    train_best_state["best_step"] = int(gstep)
    best_path = run_dir / f"best_adapter_step_{gstep}"
    try:
        _save_lora_checkpoint(
            model, tokenizer, run_dir, best_path, gstep, optimizer, scaler
        )
        train_best_state["best_path"] = str(best_path)
        (run_dir / "best.json").write_text(
            json.dumps(
                {
                    "best_step": gstep,
                    "best_score": float(score),
                    "best_metric": metric_key,
                    "best_path": str(best_path),
                    "source": "train_batch",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[train_grpo] NEW TRAIN BEST: {metric_key}={score:.4f} "
            f"at step {gstep}; saved to {best_path}"
        )
    except Exception as ee:
        print(
            f"[train_grpo] Warning: failed to save train-best adapter "
            f"at step {gstep}: {ee}"
        )


def _finalize_best_adapter(run_dir: Path, final: Path) -> None:
    """Copy train/eval best checkpoint to adapter_final when available."""
    best_json = run_dir / "best.json"
    if not best_json.is_file():
        return
    try:
        meta = json.loads(best_json.read_text(encoding="utf-8"))
        best_path = Path(str(meta.get("best_path", "")))
        if not best_path.is_dir():
            return
        import shutil

        if final.is_dir():
            shutil.rmtree(final)
        shutil.copytree(best_path, final)
        print(
            f"[train_grpo] adapter_final <- best step {meta.get('best_step')} "
            f"({meta.get('best_metric')}={meta.get('best_score')})"
        )
    except Exception as ee:
        print(f"[train_grpo] Warning: could not copy best adapter to adapter_final: {ee}")


def _resolve_train_case_subset_size(args: argparse.Namespace, n_train: int) -> int:
    """Map --train-case-subset-fraction to a case count when set (see train_grpo.py)."""
    frac = float(getattr(args, "train_case_subset_fraction", 0) or 0)
    size = int(getattr(args, "train_case_subset_size", 0) or 0)
    if frac > 0:
        size = max(1, int(round(frac * n_train)))
    return size


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
            # Dual-agent rollouts require a 'theta' adapter; from_pretrained only
            # loads 'default'. Load sibling theta/ if present, else init fresh.
            if getattr(args, "dual_agent_replay", False):
                peft_cfg = getattr(model, "peft_config", {}) or {}
                if "theta" not in peft_cfg:
                    theta_dir = adapter_resume / "theta"
                    if theta_dir.is_dir():
                        model.load_adapter(str(theta_dir), adapter_name="theta")
                        print(f"[train_grpo] Loaded verifier adapter from {theta_dir}")
                    else:
                        cfg_json = adapter_resume / "adapter_config.json"
                        if not cfg_json.is_file():
                            raise RuntimeError(
                                f"Cannot add theta adapter: missing {cfg_json}"
                            )
                        raw = json.loads(cfg_json.read_text())
                        lcfg = LoraConfig(
                            r=raw.get("r", args.lora_r),
                            lora_alpha=raw.get("lora_alpha", args.lora_alpha),
                            lora_dropout=raw.get(
                                "lora_dropout", args.lora_dropout
                            ),
                            bias=raw.get("bias", "none"),
                            task_type=raw.get("task_type", "CAUSAL_LM"),
                            target_modules=list(raw.get("target_modules", [])),
                        )
                        model.add_adapter("theta", lcfg)
                        print(
                            "[train_grpo] Added fresh theta adapter for dual-agent resume"
                        )
                model.set_adapter("default")
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
    subset_frac = float(getattr(args, "train_case_subset_fraction", 0) or 0)
    subset_size = _resolve_train_case_subset_size(args, len(full_train_cases))
    args.train_case_subset_size = subset_size
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
        "train_case_subset_fraction": subset_frac,
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
    train_best_state: Dict[str, Any] = {
        "best_score": float("-inf"),
        "best_step": 0,
        "best_path": None,
        "metric_key": _train_best_metric_key(args),
    }
    if replay_disabled:
        print(
            "[train_grpo] Replay buffer DISABLED via --disable-replay: "
            "on-policy paired-action GRPO only."
        )
    resample_each_step = bool(getattr(args, "train_case_resample_each_step", False))
    subset_size = int(getattr(args, "train_case_subset_size", 0) or 0)
    format_bad_streak = 0
    format_stop_rate = float(getattr(args, "format_stop_invalid_rate", 0.2) or 0.2)
    format_stop_patience = int(getattr(args, "format_stop_patience", 2) or 0)
    try:
        for step in range(1, args.steps + 1):
            model.train()
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
            if int(args.eval_every) <= 0:
                metric_key = train_best_state["metric_key"]
                train_score = (
                    final_acc_post
                    if metric_key == "train_accuracy_post_revise"
                    else final_acc
                )
                _maybe_save_best_train_checkpoint(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    gstep=gstep,
                    score=float(train_score),
                    train_best_state=train_best_state,
                    optimizer=optimizer,
                    scaler=scaler,
                )
            if format_stop_patience > 0 and invalid_filer_json_rate > format_stop_rate:
                format_bad_streak += 1
            else:
                format_bad_streak = 0
            if format_stop_patience > 0 and format_bad_streak >= format_stop_patience:
                print(
                    f"[train_grpo] FORMAT STOP at step {gstep}: "
                    f"invalid_filer_json_rate>{format_stop_rate} for "
                    f"{format_bad_streak} steps. Keeping the last checkpoint "
                    f"where the running train score was best."
                )
                try:
                    (run_dir / "early_stopped.json").write_text(
                        json.dumps(
                            {
                                "stopped_at_step": gstep,
                                "reason": "invalid_filer_json",
                                "invalid_filer_json_rate": invalid_filer_json_rate,
                                "patience": format_stop_patience,
                                "best_step": train_best_state.get("best_step"),
                                "best_score": train_best_state.get("best_score"),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                break
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
        if (run_dir / "best.json").is_file():
            _finalize_best_adapter(run_dir, final)
        else:
            _save_lora_checkpoint(model, tokenizer, run_dir, final, g_final, optimizer, scaler)
            print(f"[train_grpo] Saved final adapter + training_state (step {g_final}) to {final}")
