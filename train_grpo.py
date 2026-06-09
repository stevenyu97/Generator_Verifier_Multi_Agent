#!/usr/bin/env python3
"""DPA-GRPO training entry point.

Run from the repo root:
  conda activate llm
  python train_grpo.py --help
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

from benchmarks.registry import get_dataset_config
from benchmarks.tax.tax_case_taxonomy import CaseTaxonomyConfig
from dpa_grpo.config import configure_dataset
from dpa_grpo.training import train_loop
from dpa_grpo.paths import checkpoint_root



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
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(checkpoint_root()),
        help="Checkpoint root (default: $DPA_GRPO_CHECKPOINT_ROOT or ~/llm_artifacts/grpo_checkpoints).",
    )
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


if __name__ == '__main__':
    main()
