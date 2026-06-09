# DPA-GRPO

Dual Paired-Action GRPO for multi-agent numeric reasoning: a **Generator** (Filer),
**Verifier**, and **Approver** share one Qwen backbone with separate LoRA adapters.
Training uses per-line (or per-turn) 2-GRPO with an 8-case credit-assignment taxonomy.

## Layout

```
llm/
├── train_grpo.py       # Training CLI entry point
├── dpa_grpo/           # Training implementation package
├── core/               # Agents, prompts, rewards, evaluator
├── benchmarks/         # Dataset registry, line_eval seam, tax + convfinqa adapters
├── scripts/            # Training and eval shell scripts
├── paper/              # LaTeX, figures, plotting utilities
├── tax-calc-bench/     # TaxCalcBench submodule (data)
└── docs/               # Release notes and examples
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DPA_GRPO_CHECKPOINT_ROOT` | `/home/ubuntu/llm_artifacts/grpo_checkpoints` | Training checkpoints and eval logs |
| `TAX_MODEL_PATH` | `/home/ubuntu/models/models--Qwen--Qwen3-4B` | HuggingFace model path |

## Quickstart

```bash
cd /home/ubuntu/llm
pip install -r requirements.txt

# CPU wiring check (no GPU)
python benchmarks/convfinqa/smoke_test.py

# TaxCalcBench training
bash scripts/train_tax.sh

# ConvFinQA / FinQA
bash scripts/convfinqa/training_convfinqa.sh
bash scripts/convfinqa/training_finqa.sh

# Zero-shot baseline
python eval_zero_shot.py --dataset convfinqa \
  --model-path ~/models/models--Qwen--Qwen3-4B \
  --output-dir "$DPA_GRPO_CHECKPOINT_ROOT/zero_shot_convfinqa_4b"
```

## Datasets

- **Tax** (default): `tax-calc-bench/tax_calc_bench/ty24/test_data`
- **FinQA**: `benchmarks/convfinqa/data/finqa/`
- **ConvFinQA**: `benchmarks/convfinqa/data/convfinqa/`

See [`benchmarks/convfinqa/README.md`](benchmarks/convfinqa/README.md) for download/convert instructions.

## Imports

Use package paths directly, e.g. `from core.agents import filer_agent`,
`from benchmarks.registry import get_dataset_config`, and
`from benchmarks.line_eval import set_active_dataset`.
