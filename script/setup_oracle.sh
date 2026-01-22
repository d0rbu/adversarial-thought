#!/usr/bin/env bash
# Set up the activation oracles submodule
# This creates a separate virtual environment for the oracle dependencies
# Usage: ./script/setup_oracle.sh

set -euo pipefail

cd "$(dirname "$0")/.."

ORACLE_DIR="activation_oracles"

if [ ! -d "$ORACLE_DIR" ]; then
    echo "Error: activation_oracles directory not found."
    echo "Please clone/init the submodule first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "📦 Setting up activation oracles environment..."
cd "$ORACLE_DIR"

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .

echo "✅ Activation oracles environment set up!"
echo ""
echo "To use the oracle, activate its environment:"
echo "  source activation_oracles/.venv/bin/activate"
echo ""
echo "Then run oracle scripts from the activation_oracles directory."
