#!/usr/bin/env bash
# Visualize baseline vs SFT lm-eval-harness results
# Usage: ./script/viz_lmeval_baseline_vs_sft.sh

set -euo pipefail

cd "$(dirname "$0")/.."

shopt -s nullglob

BASELINE_YAMLS=(out/eval_baseline/*.yaml out/eval_baseline_quick/*.yaml)
SFT_YAMLS=(out/eval_sft/*.yaml out/eval_sft_quick/*.yaml)

if [[ ${#BASELINE_YAMLS[@]} -eq 0 ]]; then
    echo "No baseline lm-eval YAML found in out/eval_baseline*/" >&2
    exit 1
fi
if [[ ${#SFT_YAMLS[@]} -eq 0 ]]; then
    echo "No SFT lm-eval YAML found in out/eval_sft*/" >&2
    exit 1
fi

uv run python -m viz.plot_lmeval_grouped \
    "${BASELINE_YAMLS[0]}" \
    "${SFT_YAMLS[0]}" \
    --labels "Baseline,SFT" \
    --output-dir fig \
    --output-prefix baseline_vs_sft
