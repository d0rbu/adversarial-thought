#!/usr/bin/env bash
# Run activation oracle evaluation
# Usage: ./script/run_oracle.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle.sh
#   ./script/run_oracle.sh oracle=sft
#   ./script/run_oracle.sh oracle.target_adapter_path=out/my_model

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.run_oracle "$@"
