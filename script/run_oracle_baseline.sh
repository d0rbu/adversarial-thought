#!/usr/bin/env bash
# Run oracle evaluation on base model (no finetuning)
# Usage: ./script/run_oracle_baseline.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_baseline.sh
#   ./script/run_oracle_baseline.sh questions.n_questions=3
#   ./script/run_oracle_baseline.sh wandb.enabled=false
#   ./script/run_oracle_baseline.sh judge.max_tokens=500

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=default \
    experiment.name=oracle_baseline \
    experiment.output_dir=out/oracle_baseline \
    "$@"
