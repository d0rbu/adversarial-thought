#!/usr/bin/env bash
# Plot Pareto frontier (oracle vs lm-eval) for the adversarial alpha sweep.
# Expects oracle and lm-eval artifacts under out/ from run_oracle_baseline.sh,
# run_oracle_sft.sh, run_oracle_alpha_sweep.sh, run_eval_baseline.sh,
# run_eval_sft.sh, and run_eval_alpha_sweep.sh. Alpha runs without lm-eval
# are shown as dashed vertical lines at their oracle score.
#
# Usage: ./script/viz_pareto_alpha_sweep.sh [extra args for plot_pareto_alpha_sweep...]
#
# Examples:
#   ./script/viz_pareto_alpha_sweep.sh
#   ./script/viz_pareto_alpha_sweep.sh --output fig/my_pareto.png
#   ./script/viz_pareto_alpha_sweep.sh --title "My sweep"

set -euo pipefail

cd "$(dirname "$0")/.."

# Alpha values (must match run_oracle_alpha_sweep.sh / run_eval_alpha_sweep.sh)
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)
IFS=','; ALPHAS_CSV="${ALPHA_VALUES[*]}"; unset IFS

uv run python -m viz.plot_pareto_alpha_sweep \
    --out-root out \
    --alphas "$ALPHAS_CSV" \
    --output fig/pareto_alpha_sweep.png \
    "$@"
