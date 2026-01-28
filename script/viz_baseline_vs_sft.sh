#!/usr/bin/env bash
# Visualize baseline vs SFT finetuned oracle results
# Usage: ./script/viz_baseline_vs_sft.sh

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m viz.plot_score_dist_grouped \
    out/oracle_baseline/oracle_results.yaml \
    out/oracle_sft/oracle_results.yaml \
    --labels "Baseline,SFT" \
    --output fig/baseline_vs_sft.png
