#!/usr/bin/env bash
# DPA-GRPO training on FinQA (single-turn; one turn per case).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-finqa_4b}"
RUN_STEPS="${RUN_STEPS:-30}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="grpo_checkpoints/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists." >&2
  exit 1
fi

python train_grpo.py \
  --dataset finqa \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir grpo_checkpoints \
  --model-path "${TAX_MODEL_PATH:-/home/ubuntu/models/models--Qwen--Qwen3-4B}" \
  --cases-root "${HOME}/llm/convfinqa/data/finqa/train" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --train-case-subset-size 10 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --early-stop-patience 2 \
  --early-stop-min-step 10 \
  --best-metric post_revise \
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
  --alternate-rounds \
  --gradient-checkpointing \
  --amp \
  --dual-agent-replay \
  --disable-replay \
  --replay-warmup 95 \
  --linewise-rollout \
  --case-taxonomy \
  --line-case-taxonomy
