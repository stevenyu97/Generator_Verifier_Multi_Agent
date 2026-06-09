"""DPA-GRPO: eval module."""
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

from dpa_grpo.dual_rollout import _dual_linewise_rollout_transitions
from dpa_grpo.config import _gold_path
from benchmarks.line_eval import evaluated_line_ids
from core.rewards import reward_filer_draft



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
