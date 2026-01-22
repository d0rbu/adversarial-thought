#!/usr/bin/env bash
# Quick test of oracle evaluation (minimal questions/contexts)
# Usage: ./script/run_oracle_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_quick.sh
#   ./script/run_oracle_quick.sh oracle=sft
#   ./script/run_oracle_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick \
    experiment.name=oracle_quick \
    experiment.output_dir=out/oracle_quick \
    "$@"
