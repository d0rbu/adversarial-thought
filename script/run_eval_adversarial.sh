#!/usr/bin/env bash
# Evaluate adversarially trained model with lm-eval benchmarks
# Usage: ./script/run_eval_adversarial.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_adversarial.sh
#   ./script/run_eval_adversarial.sh eval.limit=100
#   ./script/run_eval_adversarial.sh eval.peft_adapter_path=out/my_custom_adversarial_adapter
#   ./script/run_eval_adversarial.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=sft \
    model.load_in_8bit=false \
    eval.peft_adapter_path=out/adversarial_sft \
    experiment.name=eval_adversarial \
    experiment.output_dir=out/eval_adversarial \
    "$@"
