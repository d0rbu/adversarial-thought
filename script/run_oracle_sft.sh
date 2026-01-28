#!/usr/bin/env bash
# Run oracle evaluation on SFT finetuned model
# Usage: ./script/run_oracle_sft.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_sft.sh
#   ./script/run_oracle_sft.sh questions.n_questions=3
#   ./script/run_oracle_sft.sh oracle.target_adapter_path=out/my_custom_adapter
#   ./script/run_oracle_sft.sh wandb.enabled=false
#   ./script/run_oracle_sft.sh judge.max_tokens=500

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=sft \
    model.load_in_8bit=false \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_sft \
    experiment.output_dir=out/oracle_sft \
    questions.split=val \
    "$@"
