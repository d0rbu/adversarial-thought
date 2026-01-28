#!/usr/bin/env bash
# Visualize baseline vs adversarial finetuned oracle results
# Usage: ./script/viz_baseline_vs_adversarial.sh

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m viz.plot_score_dist_grouped \
    out/oracle_baseline/oracle_results.yaml \
    out/oracle_adversarial/oracle_results.yaml \
    --labels "Baseline,Adversarial" \
    --output fig/baseline_vs_adversarial.png
