#!/usr/bin/env bash
# Run ConvFinQA and FinQA DPA-GRPO training.
set -euo pipefail
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
cd "$(dirname "$0")/../.."

MODEL_4B="${TAX_MODEL_PATH:-$HOME/models/Qwen3-4B}"

echo "=== DPA-GRPO training ==="
RUN_NAME="${CONVFINQA_RUN_NAME:-convfinqa_4b}" TAX_MODEL_PATH="${MODEL_4B}" \
  bash scripts/convfinqa/training_convfinqa.sh
RUN_NAME="${FINQA_RUN_NAME:-finqa_4b}" TAX_MODEL_PATH="${MODEL_4B}" \
  bash scripts/convfinqa/training_finqa.sh

echo "[done] Check ${DPA_GRPO_CHECKPOINT_ROOT} for checkpoints and eval logs"
