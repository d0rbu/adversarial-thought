#!/usr/bin/env bash
# Quick test of oracle evaluation with Qwen3-8B in 8-bit mode
# Usage: ./script/run_oracle_quick_4b.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_quick_4b.sh
#   ./script/run_oracle_quick_4b.sh oracle=sft
#   ./script/run_oracle_quick_4b.sh wandb.enabled=false
#   ./script/run_oracle_quick_4b.sh oracle.oracle_path=path/to/oracle
#
# Note: Uses Qwen3-8B model with the corresponding activation oracle checkpoint.
#       Check: https://huggingface.co/collections/adamkarvonen/activation-oracles

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick \
    model=qwen3_8b \
    oracle.model_name=Qwen/Qwen3-8B \
    oracle.oracle_path=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_quick_8b \
    experiment.output_dir=out/oracle_quick_8b \
    "$@"
