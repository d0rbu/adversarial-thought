#!/usr/bin/env bash
# Quick test of oracle evaluation on adversarially trained model (minimal questions/contexts)
# Usage: ./script/run_oracle_adversarial_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_adversarial_quick.sh
#   ./script/run_oracle_adversarial_quick.sh oracle=quick_dataset  # Use dataset contexts
#   ./script/run_oracle_adversarial_quick.sh oracle=sft
#   ./script/run_oracle_adversarial_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=quick \
    oracle.target_adapter_path=out/adversarial_sft \
    model.load_in_8bit=false \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_adversarial_quick \
    experiment.output_dir=out/oracle_adversarial_quick \
    questions.split=val \
    "$@"
