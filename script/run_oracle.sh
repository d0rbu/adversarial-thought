#!/usr/bin/env bash
# Run activation oracle evaluation
# NOTE: Must be run from the activation_oracles environment!
# Usage: ./script/run_oracle.sh [hydra overrides...]
# Examples:
#   ./script/run_oracle.sh
#   ./script/run_oracle.sh oracle=sft
#   ./script/run_oracle.sh oracle.target_adapter_path=out/my_model

set -euo pipefail

cd "$(dirname "$0")/.."

# Check if we're in the oracle environment
if [[ ! "$VIRTUAL_ENV" == *"activation_oracles"* ]]; then
    echo "⚠️  Warning: You may not be in the activation_oracles environment."
    echo "   Run: source activation_oracles/.venv/bin/activate"
    echo ""
fi

python -m exp.run_oracle "$@"
