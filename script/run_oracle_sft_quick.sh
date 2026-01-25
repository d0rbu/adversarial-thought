#!/usr/bin/env bash
# Quick test of oracle evaluation (minimal questions/contexts)
# Uses Qwen3-8B by default
# Usage: ./script/run_oracle_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_quick.sh
#   ./script/run_oracle_quick.sh oracle=quick_dataset  # Use dataset contexts
#   ./script/run_oracle_quick.sh oracle=sft
#   ./script/run_oracle_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick \
    oracle.target_adapter_path=out/sft_baseline \
    model.load_in_8bit=false \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_quick \
    experiment.output_dir=out/oracle_quick \
    "$@"
