#!/usr/bin/env bash
# Visualize baseline, SFT, and adversarial oracle results together
# Usage: ./script/viz_all_three.sh

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m viz.plot_score_dist_grouped \
    out/oracle_baseline/oracle_results.yaml \
    out/oracle_sft/oracle_results.yaml \
    out/oracle_adversarial/oracle_results.yaml \
    --labels "Baseline,SFT,Adversarial" \
    --output fig/all_three.png
