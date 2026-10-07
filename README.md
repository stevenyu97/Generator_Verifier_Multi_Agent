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
| `TAX_MODEL_PATH` | `$HOME/models/Qwen3-4B` | Local Qwen model directory |
| `CASES_ROOT` | dataset default below | Override the case folder for one training script |

## Qwen models

Weights are not in this repo. Download them with the Hugging Face CLI (`huggingface_hub` comes in with `transformers` from `requirements.txt`):

```bash
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-4B --local-dir "$HOME/models/Qwen3-4B"
export TAX_MODEL_PATH="$HOME/models/Qwen3-4B"
```

`scripts/train_tax.sh` uses Qwen3-8B when `TAX_MODEL_PATH` is unset. Download that checkpoint the same way if you want it:

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir "$HOME/models/Qwen3-8B"
export TAX_MODEL_PATH="$HOME/models/Qwen3-8B"
```

`TAX_MODEL_PATH` can be that folder, or a Hugging Face cache directory named like `models--Qwen--Qwen3-4B` that contains `snapshots/`. Training does not download the model for you.

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

That creates the folders the FinQA and ConvFinQA launchers use by default:

- `benchmarks/convfinqa/data/finqa/train`
- `benchmarks/convfinqa/data/convfinqa/train`

`.gitignore` excludes `benchmarks/convfinqa/data/` and `benchmarks/convfinqa/raw/`, so those directories will not appear on GitHub. Small checked-in samples are in `benchmarks/convfinqa/fixtures/`. Set `CASES_ROOT` to point a launcher at a different folder.

The tax adapter is [`benchmarks/tax/`](benchmarks/tax/). `scripts/train_tax.sh` reads cases from `tax-calc-bench/tax_calc_bench/ty24/test_data` under the repo root, or from `CASES_ROOT`. TaxCalcBench is not stored in this repo.

## Imports

Use package paths directly, e.g. `from core.agents import filer_agent`,
`from benchmarks.registry import get_dataset_config`, and
`from benchmarks.line_eval import set_active_dataset`.
