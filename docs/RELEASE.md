# Cutting a supplementary release tarball

The old `dpa_grpo_release/` snapshot has been removed. To publish a clean
supplementary package, bundle these directories:

```
dpa_grpo/          # Training implementation
core/              # Agents, prompts, schemas, rewards
benchmarks/        # Dataset adapters (tax + convfinqa; exclude raw/ and data/ if large)
scripts/           # train_tax.sh, eval scripts
requirements.txt
README.md
train_grpo.py      # Training CLI entry point
eval_zero_shot.py
```

**Exclude** from the tarball:
- `grpo_checkpoints/` / `$DPA_GRPO_CHECKPOINT_ROOT` (training artifacts)
- `benchmarks/convfinqa/raw/` and `benchmarks/convfinqa/data/` (users run `download_data.py` + `convert.py`)
- `tax-calc-bench/` (point users to the upstream submodule)
- `paper/` (unless publishing figures alongside)
- `train_grpo.py.bak` and other dev artifacts

**Minimal runnable release test:**
```bash
pip install -r requirements.txt
python benchmarks/convfinqa/smoke_test.py
python train_grpo.py --help
```
