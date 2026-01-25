#!/usr/bin/env bash
# Quick test of oracle evaluation (minimal questions/contexts)
# Uses Gemma 3 1B by default
# Usage: ./script/run_oracle_quick_gemma.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_quick_gemma.sh
#   ./script/run_oracle_quick_gemma.sh oracle=quick_dataset  # Use dataset contexts
#   ./script/run_oracle_quick_gemma.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick \
    model=gemma3_1b \
    oracle.model_name=google/gemma-3-1b-it \
    oracle.oracle_path=adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it \
    model.load_in_8bit=false \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_quick_gemma \
    experiment.output_dir=out/oracle_quick_gemma \
    "$@"
