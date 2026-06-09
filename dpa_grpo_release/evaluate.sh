#!/usr/bin/env bash
# Evaluate either a zero-shot baseline or a trained DPA-GRPO checkpoint.
#
# Zero-shot baseline:
#   bash evaluate.sh
#
# Trained checkpoint (e.g. best.json or adapter_final from a run dir):
#   ADAPTER=grpo_checkpoints/dpa_grpo_run/adapter_final \
#   OUT=grpo_checkpoints/dpa_grpo_run/eval_dpa_grpo \
#     bash evaluate.sh

set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="${TAX_MODEL_PATH:-Qwen/Qwen3-8B}"
CASES_ROOT="${CASES_ROOT:-./tax-calc-bench/tax_calc_bench/ty24/test_data}"
ADAPTER="${ADAPTER:-}"
OUT="${OUT:-grpo_checkpoints/$( [[ -n "${ADAPTER}" ]] && echo eval_trained || echo eval_zero_shot )}"

python evaluate.py \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER}" \
  --output-dir "${OUT}" \
  --cases-root "${CASES_ROOT}" \
  --test-fraction 0.2 \
  --split-seed 42 \
  --eval-max-cases 8 \
  --eval-temperature 0.0 \
  --eval-top-p 1.0 \
  --filer-max-new-tokens 2048 \
  --verifier-max-new-tokens 2048 \
  --revise-sample-k 5 \
  --revise-temperature 0.7
