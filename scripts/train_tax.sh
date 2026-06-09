#!/usr/bin/env bash
# Launch a GRPO training run.
#
# Usage:
#   bash run_train_grpo_first_run.sh                           # use defaults below
#   RUN_NAME=fourth_run RUN_STEPS=30 bash run_train_grpo_first_run.sh
#   RUN_NAME=fifth_run RESUME_FROM=${CKPT_ROOT}/fourth_run bash run_train_grpo_first_run.sh
#   FORCE=1 bash run_train_grpo_first_run.sh                   # overwrite existing run folder
#
# GPU (if shared): CUDA_VISIBLE_DEVICES=0 or TAX_DEVICE=cuda:0 before the command.
# CUDA debug:       CUDA_LAUNCH_BLOCKING=1 for accurate stack traces.

set -euo pipefail
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
CKPT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT}"
cd "$(dirname "$0")/.."

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-fourth_run}"
RUN_STEPS="${RUN_STEPS:-30}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="${CKPT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists." >&2
  echo "  Pick a new RUN_NAME, set RESUME_FROM=${RUN_DIR}, or set FORCE=1." >&2
  exit 1
fi
echo "[run.sh] RUN_NAME=${RUN_NAME} RUN_STEPS=${RUN_STEPS} RESUME_FROM=${RESUME_FROM:-<none>}"

python train_grpo.py \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir ${CKPT_ROOT} \
  --model-path "${TAX_MODEL_PATH:-/home/ubuntu/models/models--Qwen--Qwen3-8B}" \
  --cases-root "${HOME}/llm/tax-calc-bench/tax_calc_bench/ty24/test_data" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --train-case-subset-size 10 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --approver-balance-strength 1.5 \
  --approver-target-revise-rate 0.30 \
  --approver-rare-case-weight 1.0 \
  --approver-cave-penalty 0.2 \
  --approver-miss-penalty 0.3 \
  --approver-explore-eps 0.4 \
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

METRICS_PATH="${RUN_DIR}/metrics.jsonl"
if [[ -f "${METRICS_PATH}" ]]; then
  python paper/equilibrium_metrics.py \
    --metrics-jsonl "${METRICS_PATH}" \
    --window-size 5 \
    | tee "${RUN_DIR}/equilibrium_report.txt"
else
  echo "equilibrium report skipped: ${METRICS_PATH} not found"
fi
