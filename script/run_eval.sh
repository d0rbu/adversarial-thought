#!/usr/bin/env bash
# General evaluation script (uses Qwen3-8B by default)
# Usage: ./script/run_eval.sh [hydra overrides...]
# Examples:
#   ./script/run_eval.sh                              # Use default eval config
#   ./script/run_eval.sh eval=baseline                # Evaluate base model
#   ./script/run_eval.sh eval=sft                     # Evaluate SFT model
#   ./script/run_eval.sh eval=quick                   # Quick test evaluation
#   ./script/run_eval.sh eval.limit=100 wandb.enabled=false

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.evaluate \
    model.load_in_8bit=false \
    "$@"
