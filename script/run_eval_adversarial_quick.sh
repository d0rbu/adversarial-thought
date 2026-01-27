#!/usr/bin/env bash
# Quick evaluation of adversarially trained model (limited samples for fast iteration)
# Usage: ./script/run_eval_adversarial_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_adversarial_quick.sh
#   ./script/run_eval_adversarial_quick.sh eval.limit=50
#   ./script/run_eval_adversarial_quick.sh eval.peft_adapter_path=out/my_custom_adversarial_adapter
#   ./script/run_eval_adversarial_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=quick \
    model.load_in_8bit=false \
    eval.peft_adapter_path=out/adversarial_sft \
    experiment.name=eval_adversarial_quick \
    experiment.output_dir=out/eval_adversarial_quick \
    "$@"
