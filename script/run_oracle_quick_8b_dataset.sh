#!/usr/bin/env bash
# Quick test of oracle evaluation with Qwen3-8B using dataset data
# Usage: ./script/run_oracle_quick_8b_dataset.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_quick_8b_dataset.sh
#   ./script/run_oracle_quick_8b_dataset.sh wandb.enabled=false
#   ./script/run_oracle_quick_8b_dataset.sh oracle.oracle_path=path/to/oracle
#   ./script/run_oracle_quick_8b_dataset.sh oracle.context.n_samples=50
#
# Note: Uses Qwen3-8B model with the corresponding activation oracle checkpoint.
#       Check: https://huggingface.co/collections/adamkarvonen/activation-oracles
#       Uses dataset data from Dolci-Instruct-SFT validation split instead of custom prompts.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick_dataset \
    model=qwen3_8b \
    oracle.model_name=Qwen/Qwen3-8B \
    oracle.oracle_path=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_quick_8b_dataset \
    experiment.output_dir=out/oracle_quick_8b_dataset \
    "$@"
