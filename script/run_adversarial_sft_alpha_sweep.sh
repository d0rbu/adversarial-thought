#!/usr/bin/env bash
# Run adversarial SFT finetuning experiment with alpha sweep
# Sweeps through different values of training.adversarial.alpha
# Usage: ./script/run_adversarial_sft_alpha_sweep.sh [hydra overrides...]
# Examples:
#   ./script/run_adversarial_sft_alpha_sweep.sh
#   ./script/run_adversarial_sft_alpha_sweep.sh training.batch_size=2 training.num_epochs=3
#   ./script/run_adversarial_sft_alpha_sweep.sh wandb.enabled=false
#   ./script/run_adversarial_sft_alpha_sweep.sh data.max_samples=1000

set -euo pipefail

cd "$(dirname "$0")/.."

# Alpha values to sweep through
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)

echo "Starting alpha sweep with values: ${ALPHA_VALUES[*]}"
echo "Total runs: ${#ALPHA_VALUES[@]}"
echo ""

# Loop through each alpha value
for alpha in "${ALPHA_VALUES[@]}"; do
    # Convert alpha to a string suitable for experiment names (replace . with _)
    alpha_str=$(echo "$alpha" | tr '.' '_')

    echo "=========================================="
    echo "Running experiment with alpha=$alpha"
    echo "Experiment name: adversarial_sft_alpha_${alpha_str}"
    echo "=========================================="

    uv run python -m exp.adversarial_sft \
        training=adversarial \
        training.adversarial.alpha="$alpha" \
        experiment.name=adversarial_sft_alpha_${alpha_str} \
        experiment.output_dir=out/adversarial_sft_alpha_${alpha_str} \
        "$@"

    echo ""
    echo "Completed experiment with alpha=$alpha"
    echo ""
done

echo "=========================================="
echo "Alpha sweep completed!"
echo "=========================================="
