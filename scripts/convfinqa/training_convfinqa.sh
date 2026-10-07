#!/usr/bin/env bash
# DPA-GRPO training on ConvFinQA (multi-turn).
# No holdout eval during training — best checkpoint tracked on train-batch accuracy.
set -euo pipefail
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
CKPT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT}"
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-convfinqa_dpa_v3}"
RUN_STEPS="${RUN_STEPS:-30}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists. Set FORCE=1 to overwrite." >&2
  exit 1
fi

python train_grpo.py \
  --dataset convfinqa \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir "${CKPT_ROOT}" \
  --model-path "${TAX_MODEL_PATH:-$HOME/models/Qwen3-4B}" \
  --cases-root "${CASES_ROOT:-${REPO_ROOT}/benchmarks/convfinqa/data/convfinqa/train}" \
  --test-fraction 0.1 \
  --split-seed 42 \
  --train-case-subset-fraction 0.01 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --best-metric post_revise \
  --eval-every 0 \
  --early-stop-patience 0 \
  --group-size 2 \
  --filer-max-new-tokens 2048 \
  --verifier-max-new-tokens 2048 \
  --temperature 0.2 \
  --top-p 0.9 \
  --lr 1e-5 \
  --beta-kl 0.06 \
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
  --line-case-taxonomy \
  --approver-explore-eps 0.4 \
  --approver-miss-penalty 0.3 \
  --approver-balance-strength 1.5 \
  --approver-rare-case-weight 1.0 \
  --approver-cave-penalty 0.2

if [[ -f "${RUN_DIR}/best.json" ]]; then
  echo "[convfinqa_dpa] best checkpoint:"
  cat "${RUN_DIR}/best.json"
fi
