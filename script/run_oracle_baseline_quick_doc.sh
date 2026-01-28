#!/usr/bin/env bash
# Run oracle evaluation on base model (no finetuning)
# Usage: ./script/run_oracle_baseline_quick_doc.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_baseline_quick_doc.sh
#   ./script/run_oracle_baseline_quick_doc.sh questions.n_questions=3
#   ./script/run_oracle_baseline_quick_doc.sh wandb.enabled=false
#   ./script/run_oracle_baseline_quick_doc.sh judge.max_tokens=500

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=default \
    model.load_in_8bit=false \
    oracle.context.n_samples=10 \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_baseline_quick_doc \
    experiment.output_dir=out/oracle_baseline_quick_doc \
    "$@"
