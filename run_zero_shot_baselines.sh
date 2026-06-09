#!/usr/bin/env bash
# Run zero-shot baselines for Qwen3-4B and Qwen3-8B in two passes:
#   PASS 1 (test split): 8 held-out test cases (test_fraction=0.2 against the
#     41-case TY24 pool). Used for the headline test-acc cells in
#     paper_results_filled.tex Table 1.
#   PASS 2 (training cases in TRAIN_CASES): a hand-picked subset of training
#     cases referenced by paper_results_filled.tex Table 1, so the zero-shot
#     rows can be filled in for those columns. Default = just Train-C
#     (the only one we don't already have data for).
#
# Usage:
#   bash run_zero_shot_baselines.sh                 # both passes, both backbones
#   MODELS=4b bash run_zero_shot_baselines.sh       # only 4B (both passes)
#   MODELS=8b bash run_zero_shot_baselines.sh       # only 8B (both passes)
#   PASSES=test bash run_zero_shot_baselines.sh     # only the held-out test split
#   PASSES=train bash run_zero_shot_baselines.sh    # only the train-cases pass
#   TRAIN_CASES="case1,case2" PASSES=train bash run_zero_shot_baselines.sh
#   GPU=2 bash run_zero_shot_baselines.sh           # pin to GPU 2
#
# Outputs:
#   grpo_checkpoints/zero_shot_4b/                  (PASS 1, test split)
#   grpo_checkpoints/zero_shot_8b/
#   grpo_checkpoints/zero_shot_4b_traincases/       (PASS 2, whatever's in TRAIN_CASES)
#   grpo_checkpoints/zero_shot_8b_traincases/
# NOTE: PASS 2 OVERWRITES the *_traincases dir on each run.  The metric is
# deterministic, so previously-evaluated cases are easy to reproduce by
# rolling back TRAIN_CASES.

set -euo pipefail
cd "$(dirname "$0")"

MODEL_4B="${MODEL_4B:-/home/ubuntu/models/models--Qwen--Qwen3-4B}"
MODEL_8B="${MODEL_8B:-/home/ubuntu/models/models--Qwen--Qwen3-8B}"
CASES_ROOT="${CASES_ROOT:-${HOME}/llm/tax-calc-bench/tax_calc_bench/ty24/test_data}"
MODELS="${MODELS:-4b,8b}"
PASSES="${PASSES:-test,train}"

# Storytelling training cases referenced by Table 1 in paper_results_filled.tex.
# Override TRAIN_CASES on the command line to evaluate any other subset, e.g.:
#   TRAIN_CASES="single-multiple-w2-state-records" PASSES=train bash run_zero_shot_baselines.sh
#
# Default: only the case we still need (Train-C). Train-A and Train-B were
# already evaluated in the prior pass and zero-shot is deterministic, so
# re-running them just burns GPU time. Set TRAIN_CASES yourself if you want
# to refresh all three.
TRAIN_CASES="${TRAIN_CASES:-single-w2-box12-code-a-b-alaska}"

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  echo "[zero_shot] CUDA_VISIBLE_DEVICES=${GPU}"
fi

# PASS 1: held-out test split.
run_test() {
  local tag="$1"
  local model_path="$2"
  local out="grpo_checkpoints/zero_shot_${tag}"
  mkdir -p "${out}"
  echo "[zero_shot] === ${tag} (test split, n=8) ==="
  echo "[zero_shot] model = ${model_path}"
  echo "[zero_shot] out   = ${out}"
  python eval_zero_shot.py \
    --model-path "${model_path}" \
    --output-dir "${out}" \
    --cases-root "${CASES_ROOT}" \
    --test-fraction 0.2 \
    --split-seed 42 \
    --eval-max-cases 8 \
    --eval-temperature 0.0 \
    --eval-top-p 1.0 \
    --filer-max-new-tokens 2048 \
    --verifier-max-new-tokens 2048 \
    --revise-sample-k 5 \
    --revise-temperature 0.7 \
    2>&1 | tee "${out}/run.log"
}

# PASS 2: three storytelling training cases referenced by Table 1.
run_traincases() {
  local tag="$1"
  local model_path="$2"
  local out="grpo_checkpoints/zero_shot_${tag}_traincases"
  mkdir -p "${out}"
  echo "[zero_shot] === ${tag} (3 training cases, Train-A/B/C) ==="
  echo "[zero_shot] model = ${model_path}"
  echo "[zero_shot] out   = ${out}"
  echo "[zero_shot] cases = ${TRAIN_CASES}"
  python eval_zero_shot.py \
    --model-path "${model_path}" \
    --output-dir "${out}" \
    --cases-root "${CASES_ROOT}" \
    --test-fraction 0.2 \
    --split-seed 42 \
    --eval-max-cases 8 \
    --eval-temperature 0.0 \
    --eval-top-p 1.0 \
    --filer-max-new-tokens 2048 \
    --verifier-max-new-tokens 2048 \
    --revise-sample-k 5 \
    --revise-temperature 0.7 \
    --specific-cases "${TRAIN_CASES}" \
    2>&1 | tee "${out}/run.log"
}

IFS=',' read -ra PASS_LIST  <<< "${PASSES}"
IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
for m in "${MODEL_LIST[@]}"; do
  for pass in "${PASS_LIST[@]}"; do
    case "${m,,}-${pass,,}" in
      4b-test)  run_test       "4b" "${MODEL_4B}" ;;
      8b-test)  run_test       "8b" "${MODEL_8B}" ;;
      4b-train) run_traincases "4b" "${MODEL_4B}" ;;
      8b-train) run_traincases "8b" "${MODEL_8B}" ;;
      *) echo "[zero_shot] skipping unknown combo: ${m}-${pass}" >&2 ;;
    esac
  done
done

echo ""
echo "=== Summary ==="
for tag in 4b 8b; do
  for suffix in "" "_traincases"; do
    jsonl="grpo_checkpoints/zero_shot_${tag}${suffix}/eval_metrics.jsonl"
    if [[ -f "${jsonl}" ]]; then
      python - <<PY
import json, pathlib
rec = json.loads(open("${jsonl}").read().strip().split("\n")[-1])
label = "${tag}${suffix}" or "${tag}"
print(f"  {label}: "
      f"strict={rec['test_accuracy_strict_mean']:.4f}  "
      f"post={rec['test_accuracy_strict_post_revise_mean']:.4f}  "
      f"(n={rec['n_test_cases_evaluated']})")
TRAIN_TAGS = {
    "single-w2-retirement-sick-pay-social-security-tip":          "Train-A",
    "single-multiple-w2-excess-social-security-tax-same-ein":     "Train-B",
    "single-w2-box12-code-a-b-alaska":                             "Train-C",
}
for pc in rec.get("per_case", []):
    name = pathlib.Path(pc["case_dir"]).name
    if name in TRAIN_TAGS:
        print(f"      {TRAIN_TAGS[name]:<8s}  strict={pc['final_draft_accuracy_strict']:.3f}  "
              f"post={pc['final_draft_accuracy_post_revise']:.3f}")
PY
    fi
  done
done
