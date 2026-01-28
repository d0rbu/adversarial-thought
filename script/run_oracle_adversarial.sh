#!/usr/bin/env bash
# Run oracle evaluation on adversarially trained model
# Usage: ./script/run_oracle_adversarial.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle_adversarial.sh
#   ./script/run_oracle_adversarial.sh questions.n_questions=10
#   ./script/run_oracle_adversarial.sh oracle.target_adapter_path=out/my_custom_adversarial_adapter
#   ./script/run_oracle_adversarial.sh wandb.enabled=false
#   ./script/run_oracle_adversarial.sh judge.max_tokens=500

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle \
    oracle=sft \
    oracle.target_adapter_path=out/adversarial_sft \
    model.load_in_8bit=false \
    hardware.dtype=bfloat16 \
    experiment.name=oracle_adversarial \
    experiment.output_dir=out/oracle_adversarial \
    questions.split=val \
    "$@"
