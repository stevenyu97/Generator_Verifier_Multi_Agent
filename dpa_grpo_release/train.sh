#!/usr/bin/env bash
# DPA-GRPO training launcher.
#
# Usage:
#   bash train.sh
#   RUN_NAME=my_run RUN_STEPS=30 bash train.sh
#   RUN_NAME=resumed RESUME_FROM=grpo_checkpoints/my_run bash train.sh
#   FORCE=1 bash train.sh        # overwrite an existing run dir

set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-dpa_grpo_run}"
RUN_STEPS="${RUN_STEPS:-30}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_DIR="grpo_checkpoints/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${RESUME_FROM}" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_DIR} already exists. Pick a new RUN_NAME, set RESUME_FROM, or FORCE=1." >&2
  exit 1
fi

python train_dpa_grpo.py \
  --run-name "${RUN_NAME}" \
  --steps "${RUN_STEPS}" \
  --resume-from "${RESUME_FROM}" \
  --output-dir grpo_checkpoints \
  --model-path "${TAX_MODEL_PATH:-Qwen/Qwen3-8B}" \
  --cases-root "${CASES_ROOT:-./tax-calc-bench/tax_calc_bench/ty24/test_data}" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --train-case-subset-size 10 \
  --train-case-resample-each-step \
  --train-all-cases-per-step \
  --group-size 2 \
  --filer-max-new-tokens 2048 \
  --verifier-max-new-tokens 2048 \
  --temperature 0.2 \
  --top-p 0.9 \
  --lr 1e-5 \
  --beta-kl 0.04 \
  --revise-sample-k 5 \
  --revise-temperature 0.7 \
  --approver-balance-strength 1.5 \
  --approver-target-revise-rate 0.30 \
  --approver-rare-case-weight 1.0 \
  --approver-cave-penalty 0.2 \
  --approver-miss-penalty 0.3 \
  --approver-explore-eps 0.4 \
  --eval-every 5 \
  --eval-max-cases 8 \
  --best-metric post_revise \
  --early-stop-patience 2 \
  --early-stop-min-step 10 \
  --save-every 10 \
  --log-every 1 \
  --gradient-checkpointing \
  --amp
