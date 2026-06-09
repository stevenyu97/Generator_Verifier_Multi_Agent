#!/usr/bin/env python3
"""Evaluate a base model or a DPA-GRPO LoRA checkpoint on the held-out test split.

Reuses the same linewise rollout and 8-bucket case histogram as the training
periodic-eval block, so paper Tables 1-3 line up across zero-shot and trained
runs.

Without --adapter-path: zero-shot baseline (no LoRA wrapping; all three
agent calls share the same base policy because _set_active_adapter is a
no-op when the model isn't a PeftModel).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore

from train_dpa_grpo import (
    _resolve_model_path,
    _run_dual_periodic_eval,
    discover_case_dirs,
    split_train_test_cases,
)


def build_eval_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Construct the args namespace the rollout helper expects.

    Defaults mirror train.sh so the eval is measured under the same Filer /
    Verifier / Approver prompting and sampling conditions used at train time.
    """
    return argparse.Namespace(
        temperature=cli.eval_temperature,
        top_p=cli.eval_top_p,
        eval_temperature=cli.eval_temperature,
        eval_top_p=cli.eval_top_p,
        eval_max_cases=cli.eval_max_cases,
        filer_max_new_tokens=cli.filer_max_new_tokens,
        verifier_max_new_tokens=cli.verifier_max_new_tokens,
        revise_sample_k=cli.revise_sample_k,
        revise_temperature=cli.revise_temperature,
        line_case_amount_tol=cli.line_case_amount_tol,
        # Reward shaping is irrelevant at eval time but the rollout reads it.
        filer_outcome_bonus=0.5,
        revise_format_penalty=0.25,
        approver_cave_penalty=0.2,
        approver_miss_penalty=0.3,
        approver_explore_eps=0.0,
        approver_balance_strength=0.0,
        approver_target_revise_rate=0.30,
        approver_rare_case_weight=0.0,
        invalid_json_reward=0.0,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True,
                   help="HuggingFace cache root or snapshot dir for the base model.")
    p.add_argument("--adapter-path", default="",
                   help="Path to a DPA-GRPO LoRA checkpoint dir (e.g. .../adapter_final). "
                   "Empty = zero-shot baseline.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write eval_metrics.jsonl.")
    p.add_argument(
        "--cases-root",
        default="./tax-calc-bench/tax_calc_bench/ty24/test_data",
        help="Path to a TaxCalcBench checkout's ty24/test_data directory.",
    )
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="MUST match training value to keep the test split aligned.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="MUST match training.")

    p.add_argument("--eval-max-cases", type=int, default=8)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)
    p.add_argument("--filer-max-new-tokens", type=int, default=2048)
    p.add_argument("--verifier-max-new-tokens", type=int, default=2048)
    p.add_argument("--revise-sample-k", type=int, default=5)
    p.add_argument("--revise-temperature", type=float, default=0.7)
    p.add_argument("--line-case-amount-tol", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--specific-cases", default=None,
        help="Comma-separated case-dir basenames to evaluate INSTEAD of the held-out test split.",
    )
    cli = p.parse_args()

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model_path = _resolve_model_path(cli.model_path)
    print(f"[evaluate] device={device} model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)

    adapter_path = (cli.adapter_path or "").strip()
    if adapter_path:
        if PeftModel is None:
            raise SystemExit("peft is required to load --adapter-path; pip install peft")
        adapter_dir = Path(adapter_path).resolve()
        if not adapter_dir.is_dir():
            raise SystemExit(f"--adapter-path not found: {adapter_dir}")
        print(f"[evaluate] loading LoRA from {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    cases_root = Path(cli.cases_root).resolve()
    cases_all = discover_case_dirs(cases_root)
    if not cases_all:
        raise SystemExit(f"No cases under {cases_root}")
    train_cases, test_cases = split_train_test_cases(
        cases_all, cli.test_fraction, cli.split_seed,
    )
    print(
        f"[evaluate] {len(cases_all)} cases -> train={len(train_cases)} "
        f"test={len(test_cases)} (test_fraction={cli.test_fraction}, seed={cli.split_seed})"
    )

    if cli.specific_cases:
        wanted = {s.strip() for s in cli.specific_cases.split(",") if s.strip()}
        by_name = {p.name: p for p in cases_all}
        missing = wanted - by_name.keys()
        if missing:
            raise SystemExit(f"--specific-cases referenced unknown cases: {sorted(missing)}")
        test_cases = [by_name[n] for n in sorted(wanted)]
        print(f"[evaluate] overriding test split with {len(test_cases)} hand-picked cases")

    if not test_cases:
        raise SystemExit("No cases to evaluate.")

    out_dir = Path(cli.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl_path = out_dir / "eval_metrics.jsonl"
    config_path = out_dir / "run_config.json"

    config_path.write_text(json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "adapter_path": adapter_path or None,
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
    }, indent=2), encoding="utf-8")
    print(f"[evaluate] config -> {config_path}")
    print(f"[evaluate] eval log -> {eval_jsonl_path}")

    args = build_eval_args(cli)

    with open(eval_jsonl_path, "w", encoding="utf-8") as eval_fp:
        _run_dual_periodic_eval(
            model=model, tokenizer=tokenizer, test_cases=test_cases,
            args=args, device=device, gstep=0, eval_fp=eval_fp,
        )

    last_line = eval_jsonl_path.read_text().strip().split("\n")[-1]
    rec = json.loads(last_line)
    label = "trained" if adapter_path else "zero-shot"
    print(f"\n=== {label} result ===")
    print(f"  model:                           {model_path}")
    if adapter_path:
        print(f"  adapter:                         {adapter_path}")
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
