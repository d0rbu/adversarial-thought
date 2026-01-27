#!/usr/bin/env bash
# Quick evaluation of base model (no finetuning, limited samples for fast iteration)
# Usage: ./script/run_eval_baseline_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_baseline_quick.sh
#   ./script/run_eval_baseline_quick.sh eval.limit=50
#   ./script/run_eval_baseline_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=quick \
    model.load_in_8bit=false \
    experiment.name=eval_baseline_quick \
    experiment.output_dir=out/eval_baseline_quick \
    "$@"
