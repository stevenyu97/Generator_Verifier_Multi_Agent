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

## Training (DPA-GRPO)

```bash
bash scripts/convfinqa/training_convfinqa.sh
# or
bash scripts/convfinqa/training_finqa.sh
```
