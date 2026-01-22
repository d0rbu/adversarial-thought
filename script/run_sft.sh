#!/usr/bin/env bash
# Run SFT finetuning experiment
# Usage: ./script/run_sft.sh [hydra overrides...]
# Examples:
#   ./script/run_sft.sh
#   ./script/run_sft.sh training.batch_size=8 training.num_epochs=5
#   ./script/run_sft.sh wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.sft_finetune "$@"
