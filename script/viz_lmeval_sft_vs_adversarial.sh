#!/usr/bin/env bash
# Visualize SFT vs adversarial lm-eval-harness results
# Usage: ./script/viz_lmeval_sft_vs_adversarial.sh

set -euo pipefail

cd "$(dirname "$0")/.."

shopt -s nullglob

SFT_YAMLS=(out/eval_sft/*.yaml out/eval_sft_quick/*.yaml)
ADVERSARIAL_YAMLS=(out/eval_adversarial/*.yaml out/eval_adversarial_quick/*.yaml)

if [[ ${#SFT_YAMLS[@]} -eq 0 ]]; then
    echo "No SFT lm-eval YAML found in out/eval_sft*/" >&2
    exit 1
fi
if [[ ${#ADVERSARIAL_YAMLS[@]} -eq 0 ]]; then
    echo "No adversarial lm-eval YAML found in out/eval_adversarial*/" >&2
    exit 1
fi

uv run python -m viz.plot_lmeval_grouped \
    "${SFT_YAMLS[0]}" \
    "${ADVERSARIAL_YAMLS[0]}" \
    --labels "SFT,Adversarial" \
    --output-dir fig \
    --output-prefix sft_vs_adversarial
