#!/usr/bin/env bash
# Evaluate SFT finetuned model
# Usage: ./script/run_eval_sft.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_sft.sh
#   ./script/run_eval_sft.sh eval.limit=100
#   ./script/run_eval_sft.sh eval.peft_adapter_path=out/my_custom_adapter
#   ./script/run_eval_sft.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=sft \
    model.load_in_8bit=false \
    experiment.name=eval_sft \
    experiment.output_dir=out/eval_sft \
    "$@"
