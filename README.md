# DPA-GRPO

Dual Paired-Action GRPO for multi-agent numeric reasoning: a **Generator** (Filer),
**Verifier**, and **Approver** share one Qwen backbone with separate LoRA adapters.
Training uses per-line (or per-turn) 2-GRPO with an 8-case credit-assignment taxonomy.

## Layout

```
├── train_grpo.py    # Training CLI
├── dpa_grpo/        # Training algorithm
├── core/            # Agents, prompts, rewards, schemas
├── benchmarks/      # Tax, FinQA, and ConvFinQA adapters
├── scripts/         # Training launchers
└── requirements.txt
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DPA_GRPO_CHECKPOINT_ROOT` | `/home/ubuntu/llm_artifacts/grpo_checkpoints` | Training checkpoints and eval logs |
| `TAX_MODEL_PATH` | `/home/ubuntu/models/models--Qwen--Qwen3-4B` | HuggingFace model path |

## Quickstart

```bash
pip install -r requirements.txt
python benchmarks/convfinqa/smoke_test.py

bash scripts/train_tax.sh
bash scripts/convfinqa/training_convfinqa.sh
bash scripts/convfinqa/training_finqa.sh
```

## Datasets

FinQA and ConvFinQA are not checked in. The adapter code is
[`benchmarks/convfinqa/`](benchmarks/convfinqa/). Build the case folders locally:

```bash
cd benchmarks/convfinqa
python download_data.py
python convert.py
```

That creates:

- `benchmarks/convfinqa/data/finqa/`
- `benchmarks/convfinqa/data/convfinqa/`

`.gitignore` excludes `benchmarks/convfinqa/data/` and `benchmarks/convfinqa/raw/`, so those directories will not appear on GitHub. Small checked-in samples are in `benchmarks/convfinqa/fixtures/`.

The tax adapter is [`benchmarks/tax/`](benchmarks/tax/). TaxCalcBench cases are external and are not stored in this repo.

## Imports

Use package paths directly, e.g. `from core.agents import filer_agent`,
`from benchmarks.registry import get_dataset_config`, and
`from benchmarks.line_eval import set_active_dataset`.
