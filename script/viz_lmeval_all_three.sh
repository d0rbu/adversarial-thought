#!/usr/bin/env bash
# Visualize baseline, SFT, and adversarial lm-eval-harness results together
# Usage: ./script/viz_lmeval_all_three.sh

set -euo pipefail

cd "$(dirname "$0")/.."

shopt -s nullglob

BASELINE_YAMLS=(out/eval_baseline/*.yaml out/eval_baseline_quick/*.yaml)
SFT_YAMLS=(out/eval_sft/*.yaml out/eval_sft_quick/*.yaml)
ADVERSARIAL_YAMLS=(out/eval_adversarial/*.yaml out/eval_adversarial_quick/*.yaml)

if [[ ${#BASELINE_YAMLS[@]} -eq 0 ]]; then
    echo "No baseline lm-eval YAML found in out/eval_baseline*/" >&2
    exit 1
fi
if [[ ${#SFT_YAMLS[@]} -eq 0 ]]; then
    echo "No SFT lm-eval YAML found in out/eval_sft*/" >&2
    exit 1
fi
if [[ ${#ADVERSARIAL_YAMLS[@]} -eq 0 ]]; then
    echo "No adversarial lm-eval YAML found in out/eval_adversarial*/" >&2
    exit 1
fi

uv run python -m viz.plot_lmeval_grouped \
    "${BASELINE_YAMLS[0]}" \
    "${SFT_YAMLS[0]}" \
    "${ADVERSARIAL_YAMLS[0]}" \
    --labels "Baseline,SFT,Adversarial" \
    --output-dir fig \
    --output-prefix all_three
