#!/usr/bin/env bash
# Filer-only DAPO baseline launcher (no Verifier, no Approver).
#
# Replaces the REINFORCE-style GRPO loss in the Filer-only branch with
# DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization,
# Yu et al. 2025). Asymmetric clip (epsilon_low=0.2, epsilon_high=0.28),
# token-level loss reduction, and dynamic sampling (already enforced by
# the std>1e-12 guard in the filer-only update). KL anchor is dropped
# (clip controls drift).
#
# Usage:
#   RUN_NAME=filer_only_dapo_4b RUN_STEPS=20 bash training_filer_only_dapo.sh
#   TAX_MODEL_PATH=/home/ubuntu/models/models--Qwen--Qwen3-8B \
#     RUN_NAME=filer_only_dapo_8b RUN_STEPS=20 bash training_filer_only_dapo.sh
#
# Companion: training_filer_only_gspo.sh (sequence-level ratio variant).

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-filer_only_dapo_8b}"
RUN_STEPS="${RUN_STEPS:-20}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="grpo_checkpoints/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists." >&2
  echo "  Pick a new RUN_NAME, set RESUME_FROM=${RUN_DIR}, or set FORCE=1." >&2
  exit 1
fi
echo "[filer_only_dapo.sh] RUN_NAME=${RUN_NAME} RUN_STEPS=${RUN_STEPS} RESUME_FROM=${RESUME_FROM:-<none>}"

python train_grpo.py \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir grpo_checkpoints \
  --model-path "${TAX_MODEL_PATH:-/home/ubuntu/models/models--Qwen--Qwen3-8B}" \
  --cases-root "${HOME}/llm/tax-calc-bench/tax_calc_bench/ty24/test_data" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --train-case-subset-size 5 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --no-multi-agent \
  --use-dapo \
  --dapo-clip-low 0.2 \
  --dapo-clip-high 0.28 \
  --early-stop-patience 3 \
  --early-stop-min-step 10 \
  --best-metric strict \
  --eval-every 5 \
  --eval-max-cases 8 \
  --group-size 2 \
  --filer-max-new-tokens 2048 \
  --verifier-max-new-tokens 2048 \
  --temperature 0.2 \
  --top-p 0.9 \
  --lr 1e-5 \
  --beta-kl 0.0 \
  --save-every 10 \
  --log-every 1 \
  --use-lora \
  --gradient-checkpointing \
  --amp \
  --dual-agent-replay \
  --disable-replay \
  --replay-warmup 95 \
  --linewise-rollout

METRICS_PATH="${RUN_DIR}/metrics.jsonl"
if [[ -f "${METRICS_PATH}" ]]; then
  echo "[filer_only_dapo.sh] metrics: ${METRICS_PATH}"
  echo "[filer_only_dapo.sh] eval:    ${RUN_DIR}/eval_metrics.jsonl"
  if [[ -f "${RUN_DIR}/best.json" ]]; then
    echo "[filer_only_dapo.sh] best checkpoint:"
    cat "${RUN_DIR}/best.json"
  fi
fi
