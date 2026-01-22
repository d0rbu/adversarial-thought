#!/usr/bin/env bash
# Quick evaluation of SFT finetuned model (limited samples for fast iteration)
# Usage: ./script/run_eval_sft_quick.sh [hydra overrides...]
# Examples:
#   ./script/run_eval_sft_quick.sh
#   ./script/run_eval_sft_quick.sh eval.limit=50
#   ./script/run_eval_sft_quick.sh eval.peft_adapter_path=out/my_custom_adapter
#   ./script/run_eval_sft_quick.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    eval=quick \
    eval.peft_adapter_path=out/sft_baseline \
    experiment.name=eval_sft_quick \
    experiment.output_dir=out/eval_sft_quick \
    "$@"
