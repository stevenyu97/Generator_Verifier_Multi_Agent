# ConvFinQA / FinQA for DPA-GRPO

This folder adapts the two-player linewise DPA-GRPO pipeline to **FinQA** (single-turn) and **ConvFinQA** (multi-turn).

## Setup

```bash
cd /home/ubuntu/llm/benchmarks/convfinqa
python download_data.py          # FinQA JSON + ConvFinQA data.zip
python convert.py                # -> data/{finqa,convfinqa}/{train,test}/
python smoke_test.py             # CPU wiring check (no GPU/torch needed)
```

Current converted sizes (after official splits):
- **FinQA**: 6,251 train / 883 test (dev) cases, 1 turn each
- **ConvFinQA**: 3,037 train / 421 test (dev) cases, variable turns

## Zero-shot baseline

```bash
cd /home/ubuntu/llm
export DPA_GRPO_CHECKPOINT_ROOT="${DPA_GRPO_CHECKPOINT_ROOT:-/home/ubuntu/llm_artifacts/grpo_checkpoints}"
python eval_zero_shot.py \
  --dataset convfinqa \
  --model-path /home/ubuntu/models/models--Qwen--Qwen3-4B \
  --output-dir "$DPA_GRPO_CHECKPOINT_ROOT/zero_shot_convfinqa_4b"
```

## Training (DPA-GRPO)

```bash
bash scripts/convfinqa/training_convfinqa.sh
# or
bash scripts/convfinqa/training_finqa.sh
```

Full pipeline when GPU is available:

```bash
bash scripts/convfinqa/run_when_gpu_ready.sh
```
