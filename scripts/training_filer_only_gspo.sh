#!/usr/bin/env bash
# Filer-only GSPO baseline launcher (no Verifier, no Approver).
#
# Replaces the REINFORCE-style GRPO loss in the Filer-only branch with
# GSPO (Group Sequence Policy Optimization, Zheng et al. 2025, Alibaba/Qwen).
# Sequence-level importance ratio - exp(mean_t(lp_new_t - lp_old_t)) -
# and a single symmetric clip. GSPO's claim is that token-level ratios
# produce high-variance updates on long sequences (and especially on
# MoE models); sequence-level ratios stabilise this. KL anchor is
# dropped (clip controls drift).
#
# Usage:
#   RUN_NAME=filer_only_gspo_4b RUN_STEPS=20 bash training_filer_only_gspo.sh
#   TAX_MODEL_PATH=/home/ubuntu/models/models--Qwen--Qwen3-8B \
#     RUN_NAME=filer_only_gspo_8b RUN_STEPS=20 bash training_filer_only_gspo.sh
#
# Companion: training_filer_only_dapo.sh (token-level + asymmetric clip variant).

set -euo pipefail
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
CKPT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT}"
cd "$(dirname "$0")/.."

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-filer_only_gspo_8b}"
RUN_STEPS="${RUN_STEPS:-20}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists." >&2
  echo "  Pick a new RUN_NAME, set RESUME_FROM=${RUN_DIR}, or set FORCE=1." >&2
  exit 1
fi
echo "[filer_only_gspo.sh] RUN_NAME=${RUN_NAME} RUN_STEPS=${RUN_STEPS} RESUME_FROM=${RESUME_FROM:-<none>}"

python train_grpo.py \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir ${CKPT_ROOT} \
  --model-path "${TAX_MODEL_PATH:-/home/ubuntu/models/models--Qwen--Qwen3-8B}" \
  --cases-root "${HOME}/llm/tax-calc-bench/tax_calc_bench/ty24/test_data" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --train-case-subset-size 5 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --no-multi-agent \
  --use-gspo \
  --gspo-clip 0.2 \
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
  echo "[filer_only_gspo.sh] metrics: ${METRICS_PATH}"
  echo "[filer_only_gspo.sh] eval:    ${RUN_DIR}/eval_metrics.jsonl"
  if [[ -f "${RUN_DIR}/best.json" ]]; then
    echo "[filer_only_gspo.sh] best checkpoint:"
    cat "${RUN_DIR}/best.json"
  fi
fi
