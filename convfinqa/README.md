# ConvFinQA / FinQA for DPA-GRPO

This folder adapts the two-player linewise DPA-GRPO pipeline to **FinQA** (single-turn) and **ConvFinQA** (multi-turn).

## Setup

```bash
cd /home/ubuntu/llm/convfinqa
python download_data.py          # FinQA JSON + ConvFinQA data.zip
python convert.py                # -> data/{finqa,convfinqa}/{train,test}/
python smoke_test.py             # CPU wiring check (no GPU/torch needed)
```

Current converted sizes (after official splits):
- **FinQA**: 6,251 train / 883 test (dev) cases, 1 turn each
- **ConvFinQA**: 3,037 train / 421 test (dev) cases, variable turns

Converted cases live under `data/{finqa,convfinqa}/{train,test}/<case_id>/` with:
- `input.json` — document + ordered questions
- `gold.json` — per-turn gold answers (`turn_0`, `turn_1`, ...)

## Zero-shot baseline

```bash
cd /home/ubuntu/llm
python eval_zero_shot.py \
  --dataset convfinqa \
  --model-path /home/ubuntu/models/models--Qwen--Qwen3-4B \
  --output-dir grpo_checkpoints/zero_shot_convfinqa_4b
```

Use `--dataset finqa` for single-turn FinQA.

## Training (DPA-GRPO)

```bash
bash convfinqa/training_convfinqa.sh
# or
bash convfinqa/training_finqa.sh
```

Both scripts call `train_grpo.py` with `--dataset` and the matching `--cases-root`.

When GPU is available, run the full baseline + training pipeline:

```bash
bash convfinqa/run_when_gpu_ready.sh
```

## Mapping to the tax pipeline

| TaxCalcBench | FinQA / ConvFinQA |
|---|---|
| `output.xml` | `gold.json` |
| 1040 line id (`1a`, `9`, ...) | turn id (`turn_0`, `turn_1`, ...) |
| exact float vs XML | tolerant numeric compare |
| 19 fixed lines / case | variable turns / case |

The core 2-GRPO loop, SAC/keep-revise dynamics, and 6/8-case taxonomy are unchanged.
