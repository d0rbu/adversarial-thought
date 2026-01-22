#!/usr/bin/env bash
# Run activation oracle evaluation
# Usage: ./script/run_oracle.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle.sh
#   ./script/run_oracle.sh oracle=sft
#   ./script/run_oracle.sh oracle.target_adapter_path=out/my_model
#
# Convenience scripts:
#   ./script/run_oracle_quick.sh      - Quick test (2 questions, 2 contexts)
#   ./script/run_oracle_baseline.sh  - Base model evaluation
#   ./script/run_oracle_sft.sh        - SFT finetuned model evaluation

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle "$@"
