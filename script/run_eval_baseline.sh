#!/usr/bin/env bash
# Evaluate base model (no finetuning)
# Usage: ./script/run_eval_baseline.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_baseline.sh
#   ./script/run_eval_baseline.sh eval.limit=100
#   ./script/run_eval_baseline.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=baseline \
    model.load_in_8bit=false \
    experiment.name=eval_baseline \
    experiment.output_dir=out/eval_baseline \
    "$@"
