#!/usr/bin/env bash
# Visualize SFT finetuned vs adversarial finetuned oracle results
# Usage: ./script/viz_sft_vs_adversarial.sh

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m viz.plot_score_dist_grouped \
    out/oracle_sft/oracle_results.yaml \
    out/oracle_adversarial/oracle_results.yaml \
    --labels "SFT,Adversarial" \
    --output fig/sft_vs_adversarial.png
