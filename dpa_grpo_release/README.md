# DPA-GRPO

Reference implementation of **Dual Paired-Action GRPO** for the Generator /
Verifier tax-filing setup used in the paper. The Generator and Verifier are
two LoRA adapters on a shared Qwen backbone; a third Approver head adjudicates
the Verifier's KEEP / REVISE recommendation per line.

## Overview

For each evaluated 1040 line `l` in a case `s`, one rollout produces:

1. `x_l ~ pi_phi(. | input, context_<l)` — filer line draft (default adapter).
2. `y_l ~ pi_theta(. | input, x_l)` — verifier action in `{SAC, NS}` (theta adapter).
3. If `y_l == SAC`: best-of-K `z_l` from `pi_phi(. | input, x_l, y_l)` and an
   approver decision `a_l in {KEEP, REVISE}` (default adapter).

The GRPO update is paired-action: every transition contributes counterfactual
`{SAC, NS}` and `{KEEP, REVISE}` losses regardless of the action sampled, so
both stubs of every decision get gradient signal each step. Reward shaping
(cave / miss penalties, ε-exploration on the approver, class balancing) is
described in §3 of the paper. See `train_dpa_grpo._dual_linewise_rollout_transitions`
and `_apply_transition_batch_update` for the implementation, which together
match Algorithm 1.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is pinned `>=2.1` (any CUDA build that matches your driver). Training
on a single 24 GB GPU with `--gradient-checkpointing --amp` is the supported
configuration. A CPU run is possible for sanity checking but is impractical
for full training.

## Data

The TaxCalcBench dataset
(<https://github.com/column-tax/tax-calc-bench>) provides per-case
`input.json` and `output.xml`. Point `--cases-root` (or `CASES_ROOT`) at
the `ty24/test_data` directory of a TaxCalcBench checkout. The
train / test split is deterministic in `--test-fraction` and `--split-seed`,
so the same split appears at training, eval, and ablation time.

## Training

```bash
RUN_NAME=dpa_grpo_run RUN_STEPS=30 bash train.sh
```

Per-step artifacts under `grpo_checkpoints/<run-name>/`:

- `metrics.jsonl` — one JSON object per step (8-bucket case histogram, SAC /
  REVISE rates, decision accuracy, per-role rewards, post-revise accuracy).
- `eval_metrics.jsonl` — periodic test-split evaluation (cadence
  `--eval-every`).
- `step_traces/step_*.json` — full per-line transitions, including filer and
  approver text and the case partition `{1, 2, 3, 4, 5a, 5b, 6a, 6b}`.
- `adapter_step_*` and `adapter_final/` — LoRA checkpoints. `best.json`
  points at the best-test-set checkpoint.

### Resuming

```bash
RUN_NAME=resumed RESUME_FROM=grpo_checkpoints/dpa_grpo_run \
  RUN_STEPS=10 bash train.sh
```

`adapter_final/` and (when present) `training_state.pt` are loaded from
`RESUME_FROM`.

## Evaluation

Zero-shot baseline:

```bash
bash evaluate.sh
```

Trained checkpoint:

```bash
ADAPTER=grpo_checkpoints/dpa_grpo_run/adapter_final \
OUT=grpo_checkpoints/dpa_grpo_run/eval_trained \
  bash evaluate.sh
```

Output goes to `<OUT>/eval_metrics.jsonl` and matches the schema produced by
the periodic-eval block during training, so plotting / ablation scripts can
consume both interchangeably.

## Reproducing the paper

The headline numbers (Table 1 strict and post-revise accuracy, Table 2 case
histogram, Table 3 decision accuracy) come from a single full run with the
flags in `train.sh`:

- backbone: `Qwen3-8B`
- 30 steps, group size 2, lr 1e-5, bf16 AMP, gradient checkpointing
- best-of-5 revise sampling at `revise_temperature=0.7`
- approver shaping: `cave_penalty=0.2`, `miss_penalty=0.3`,
  `balance_strength=1.5`, `target_revise_rate=0.30`,
  `rare_case_weight=1.0`, `explore_eps=0.4`
- early stop on `test_accuracy_strict_post_revise_mean` with `patience=2`,
  `min_step=10`

The reported test accuracy uses the checkpoint pointed to by `best.json`.
Running `bash evaluate.sh` with `ADAPTER=$(jq -r .best_path
grpo_checkpoints/dpa_grpo_run/best.json)` reproduces the row.
