#!/usr/bin/env bash
# Run ConvFinQA / FinQA baselines and DPA-GRPO once GPU + torch are available.
set -euo pipefail
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
CKPT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT}"
cd "$(dirname "$0")/.."

MODEL_4B="${TAX_MODEL_PATH:-/home/ubuntu/models/models--Qwen--Qwen3-4B}"
MODEL_8B="${TAX_MODEL_PATH_8B:-/home/ubuntu/models/models--Qwen--Qwen3-8B}"

echo "=== Zero-shot baselines ==="
python eval_zero_shot.py --dataset convfinqa --model-path "${MODEL_4B}" \
  --output-dir ${CKPT_ROOT}/zero_shot_convfinqa_4b
python eval_zero_shot.py --dataset finqa --model-path "${MODEL_4B}" \
  --output-dir ${CKPT_ROOT}/zero_shot_finqa_4b

echo "=== DPA-GRPO training (4B) ==="
RUN_NAME=convfinqa_4b TAX_MODEL_PATH="${MODEL_4B}" bash convfinqa/training_convfinqa.sh
RUN_NAME=finqa_4b TAX_MODEL_PATH="${MODEL_4B}" bash convfinqa/training_finqa.sh

echo "=== Optional 8B runs ==="
# RUN_NAME=convfinqa_8b TAX_MODEL_PATH="${MODEL_8B}" bash convfinqa/training_convfinqa.sh
# RUN_NAME=finqa_8b TAX_MODEL_PATH="${MODEL_8B}" bash convfinqa/training_finqa.sh

echo "[done] Check ${CKPT_ROOT}/*/eval_metrics.jsonl for strict/post-revise accuracy + case histogram"
