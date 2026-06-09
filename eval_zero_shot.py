#!/usr/bin/env python3
"""Zero-shot baseline eval on the held-out test cases.

Loads a base model with NO LoRA adapter and runs the same linewise eval
pipeline used during training. Reports per-case strict and post-revise
accuracy, plus the 8-bucket case histogram, identical in shape to the
periodic-eval JSONL written during training.

Why this works: train_grpo._set_active_adapter() is a no-op when the model
isn't wrapped by PEFT, so all three agent calls (Filer, Verifier, Approver)
share the same base policy — the canonical "zero-shot" setting.

Usage:
  python eval_zero_shot.py \\
    --model-path /home/ubuntu/models/models--Qwen--Qwen3-4B \\
    --output-dir grpo_checkpoints/zero_shot_4b \\
    --cases-root /home/ubuntu/llm/tax-calc-bench/tax_calc_bench/ty24/test_data

Reuses the same train/test split logic (test_fraction=0.2, split_seed=42)
so the test cases line up exactly with the trained-model eval blocks.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_registry import get_dataset_config
from train_grpo import (
    _resolve_model_path,
    _run_dual_periodic_eval,
    configure_dataset,
    discover_case_dirs,
    split_train_test_cases,
)


def build_eval_args(
    eval_max_cases: int,
    eval_temperature: float,
    eval_top_p: float,
    filer_max_new_tokens: int,
    verifier_max_new_tokens: int,
    revise_sample_k: int,
    revise_temperature: float,
    line_case_amount_tol: float,
) -> argparse.Namespace:
    """Construct the args namespace the rollout/eval helpers expect.

    Defaults mirror the training script (training.sh) so this baseline is
    measured under the same Filer/Verifier/Approver prompting and sampling
    conditions, with the only change being: no trained LoRA adapters.
    """
    return argparse.Namespace(
        # Eval-time sampling.
        temperature=eval_temperature,
        top_p=eval_top_p,
        eval_temperature=eval_temperature,
        eval_top_p=eval_top_p,
        eval_max_cases=eval_max_cases,
        # Rollout knobs.
        group_size=2,
        filer_max_new_tokens=filer_max_new_tokens,
        verifier_max_new_tokens=verifier_max_new_tokens,
        revise_sample_k=revise_sample_k,
        revise_temperature=revise_temperature,
        line_case_amount_tol=line_case_amount_tol,
        approver_cave_penalty=0.2,
        # Adapter / role assignment. With no PEFT wrapping these become no-ops
        # and all three agent calls hit the same base policy.
        swap_roles=False,
        role_swap_period=10,
        # Verifier prompt selection.
        line_case_taxonomy=True,
        case_taxonomy=True,
        # Misc flags the rollout may consult.
        invalid_json_reward=0.0,
        line_case_reward_scale=0.1,
        # Approver-improvement flags from the latest training.sh — set to 0
        # so they don't bias the zero-shot baseline.
        approver_balance_strength=0.0,
        approver_target_revise_rate=0.30,
        approver_rare_case_weight=0.0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace cache root or snapshot dir for the base model.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write eval_metrics.jsonl. Created if missing.",
    )
    p.add_argument(
        "--dataset",
        choices=["tax", "finqa", "convfinqa"],
        default="tax",
        help="Benchmark dataset.",
    )
    p.add_argument(
        "--cases-root",
        default=None,
        help="Root containing per-case directories with input.json and gold file.",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of cases held out for testing. MUST match training value to "
        "keep the test split identical.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the deterministic train/test split. MUST match training.",
    )
    p.add_argument("--eval-max-cases", type=int, default=8)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)
    p.add_argument("--filer-max-new-tokens", type=int, default=2048)
    p.add_argument("--verifier-max-new-tokens", type=int, default=2048)
    p.add_argument("--revise-sample-k", type=int, default=5)
    p.add_argument("--revise-temperature", type=float, default=0.7)
    p.add_argument("--line-case-amount-tol", type=float, default=0.5)
    p.add_argument(
        "--device",
        default=None,
        help="Override device (cuda, cuda:0, cpu, ...). Defaults to cuda if available.",
    )
    p.add_argument(
        "--specific-cases",
        default=None,
        help="Comma-separated list of case-directory basenames to evaluate "
        "INSTEAD of the held-out test split. Useful to evaluate the "
        "zero-shot baseline on a hand-picked subset of training cases.",
    )
    cli = p.parse_args()
    configure_dataset(cli.dataset)
    if not cli.cases_root:
        cli.cases_root = str(get_dataset_config(cli.dataset).default_eval_cases_root)

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model_path = _resolve_model_path(cli.model_path)
    print(f"[zero_shot] Loading tokenizer/model from {model_path}")
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
    model.eval()

    cases_root = Path(cli.cases_root).resolve()
    cases_all = discover_case_dirs(cases_root)
    if not cases_all:
        raise SystemExit(f"No cases under {cases_root}")
    train_cases, test_cases = split_train_test_cases(
        cases_all, cli.test_fraction, cli.split_seed
    )
    print(
        f"[zero_shot] Found {len(cases_all)} cases; "
        f"train={len(train_cases)} test={len(test_cases)} "
        f"(test_fraction={cli.test_fraction}, split_seed={cli.split_seed})"
    )

    # Override the evaluation set with a hand-picked list when requested.
    # Useful for zero-shot baselines on specific training cases that show
    # up in the headline results table.
    if cli.specific_cases:
        wanted = {s.strip() for s in cli.specific_cases.split(",") if s.strip()}
        by_name = {p.name: p for p in cases_all}
        missing = wanted - by_name.keys()
        if missing:
            raise SystemExit(
                f"--specific-cases referenced unknown cases: {sorted(missing)}\n"
                f"Available cases under {cases_root} ({len(by_name)} total):\n  "
                + "\n  ".join(sorted(by_name)[:25]) + "\n  ..."
            )
        test_cases = [by_name[name] for name in sorted(wanted)]
        print(f"[zero_shot] Overriding test split with {len(test_cases)} hand-picked cases:")
        for c in test_cases:
            print(f"   {c.name}")

    if not test_cases:
        raise SystemExit("No cases to evaluate (test split empty and no --specific-cases).")

    out_dir = Path(cli.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl_path = out_dir / "eval_metrics.jsonl"
    config_path = out_dir / "run_config.json"

    config_path.write_text(
        json.dumps(
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": model_path,
                "dataset": cli.dataset,
                "cases_root": str(cases_root),
                "test_fraction": cli.test_fraction,
                "split_seed": cli.split_seed,
                "n_test_cases": len(test_cases),
                "test_case_dirs": [str(p.resolve()) for p in test_cases],
                "eval_max_cases": cli.eval_max_cases,
                "eval_temperature": cli.eval_temperature,
                "eval_top_p": cli.eval_top_p,
                "revise_sample_k": cli.revise_sample_k,
                "revise_temperature": cli.revise_temperature,
                "use_lora": False,
                "note": "Zero-shot baseline: base model with no LoRA adapter. "
                "Filer/Verifier/Approver all share the same base policy.",
            },
            indent=2,
        )
    )
    print(f"[zero_shot] Config -> {config_path}")
    print(f"[zero_shot] Eval log -> {eval_jsonl_path}")

    args = build_eval_args(
        eval_max_cases=cli.eval_max_cases,
        eval_temperature=cli.eval_temperature,
        eval_top_p=cli.eval_top_p,
        filer_max_new_tokens=cli.filer_max_new_tokens,
        verifier_max_new_tokens=cli.verifier_max_new_tokens,
        revise_sample_k=cli.revise_sample_k,
        revise_temperature=cli.revise_temperature,
        line_case_amount_tol=cli.line_case_amount_tol,
    )

    with open(eval_jsonl_path, "w", encoding="utf-8") as eval_fp:
        _run_dual_periodic_eval(
            model=model,
            tokenizer=tokenizer,
            test_cases=test_cases,
            args=args,
            device=device,
            gstep=0,
            eval_fp=eval_fp,
        )

    last_line = eval_jsonl_path.read_text().strip().split("\n")[-1]
    rec = json.loads(last_line)
    print("\n=== Zero-shot baseline result ===")
    print(f"  model: {model_path}")
    print(f"  n_test_cases:                    {rec['n_test_cases_evaluated']}")
    print(f"  test_accuracy_strict_mean:       {rec['test_accuracy_strict_mean']:.4f}")
    print(f"  test_accuracy_post_revise_mean:  {rec['test_accuracy_strict_post_revise_mean']:.4f}")
    print(f"  test_line_accuracy_mean:         {rec['test_line_accuracy_mean']:.4f}")
    print(f"  test_line_accuracy_post_revise:  {rec['test_line_accuracy_post_revise_mean']:.4f}")
    print(f"  case_histogram:                  {rec['test_case_histogram']}")
    print("\nPer-case:")
    for pc in rec["per_case"]:
        cd = pc["case_dir"].split("/")[-1]
        print(
            f"  {cd:<55s} strict={pc['final_draft_accuracy_strict']:.3f}  "
            f"post={pc['final_draft_accuracy_post_revise']:.3f}"
        )


if __name__ == "__main__":
    main()
