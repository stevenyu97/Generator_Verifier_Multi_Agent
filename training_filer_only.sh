#!/usr/bin/env bash
# Filer-only GRPO baseline launcher (no Verifier, no Approver).
#
# Trains the Filer adapter via paired-action GRPO on K=2 sampled drafts
# per evaluated 1040 line, with reward = oracle line correctness S_x.
# Provides the missing "trained-Filer-only" control row for the paper's
# main results table -- isolates the contribution of the multi-agent loop.
#
# Usage:
#   RUN_NAME=filer_only_4b RUN_STEPS=20 bash training_filer_only.sh
#   TAX_MODEL_PATH=/home/ubuntu/models/models--Qwen--Qwen3-8B \
#     RUN_NAME=filer_only_8b RUN_STEPS=20 bash training_filer_only.sh
#
# Defaults to Qwen3-4B because the 4B Filer has more headroom than the 8B
# zero-shot baseline (0.414 -> ~0.467 in the multi-agent runs); a Filer-only
# 4B baseline tells us how much of that gain came from the loop vs the GRPO
# signal alone.

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-filer_only_4b}"
RUN_STEPS="${RUN_STEPS:-20}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="grpo_checkpoints/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists." >&2
  echo "  Pick a new RUN_NAME, set RESUME_FROM=${RUN_DIR}, or set FORCE=1." >&2
  exit 1
fi
echo "[filer_only.sh] RUN_NAME=${RUN_NAME} RUN_STEPS=${RUN_STEPS} RESUME_FROM=${RESUME_FROM:-<none>}"

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
  --beta-kl 0.04 \
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
  echo "[filer_only.sh] metrics: ${METRICS_PATH}"
  echo "[filer_only.sh] eval:    ${RUN_DIR}/eval_metrics.jsonl"
  if [[ -f "${RUN_DIR}/best.json" ]]; then
    echo "[filer_only.sh] best checkpoint:"
    cat "${RUN_DIR}/best.json"
  fi
fi
