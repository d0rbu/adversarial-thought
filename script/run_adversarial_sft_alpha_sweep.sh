#!/usr/bin/env bash
# Run adversarial SFT finetuning experiment with alpha sweep
# Sweeps through different values of training.adversarial.alpha
# Usage: ./script/run_adversarial_sft_alpha_sweep.sh [--start-from ALPHA] [hydra overrides...]
# Examples:
#   ./script/run_adversarial_sft_alpha_sweep.sh
#   ./script/run_adversarial_sft_alpha_sweep.sh --start-from 0.004
#   ./script/run_adversarial_sft_alpha_sweep.sh training.batch_size=2 training.num_epochs=3
#   ./script/run_adversarial_sft_alpha_sweep.sh --start-from 0.01 wandb.enabled=false
#   ./script/run_adversarial_sft_alpha_sweep.sh data.max_samples=1000

set -euo pipefail

cd "$(dirname "$0")/.."

# Alpha values to sweep through
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)

# Parse --start-from option if provided
START_FROM=""
HYDRA_ARGS=()
SKIP_MODE="auto"  # "auto" = skip existing, "none" = don't skip

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --no-skip)
            SKIP_MODE="none"
            shift
            ;;
        *)
            HYDRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Find starting index if --start-from is specified
START_INDEX=0
if [[ -n "$START_FROM" ]]; then
    for i in "${!ALPHA_VALUES[@]}"; do
        if [[ "${ALPHA_VALUES[$i]}" == "$START_FROM" ]]; then
            START_INDEX=$i
            echo "Resuming from alpha=$START_FROM (index $i)"
            break
        fi
    done
    if [[ $START_INDEX -eq 0 && "${ALPHA_VALUES[0]}" != "$START_FROM" ]]; then
        echo "Warning: --start-from value $START_FROM not found in alpha values. Starting from beginning."
        START_INDEX=0
    fi
fi

echo "Starting alpha sweep with values: ${ALPHA_VALUES[*]}"
echo "Starting from index: $START_INDEX (alpha=${ALPHA_VALUES[$START_INDEX]})"
echo "Total runs: ${#ALPHA_VALUES[@]}"
echo "Skip mode: $SKIP_MODE"
echo ""

# Loop through each alpha value starting from START_INDEX
for ((i=$START_INDEX; i<${#ALPHA_VALUES[@]}; i++)); do
    alpha="${ALPHA_VALUES[$i]}"
    # Convert alpha to a string suitable for experiment names (replace . with _)
    alpha_str=$(echo "$alpha" | tr '.' '_')
    output_dir="out/adversarial_sft_alpha_${alpha_str}"

    # Check if output directory exists and is non-empty (has adapter_config.json) in auto mode
    if [[ "$SKIP_MODE" == "auto" && -d "$output_dir" && -f "$output_dir/adapter_config.json" && -s "$output_dir/adapter_config.json" ]]; then
        echo "=========================================="
        echo "Skipping alpha=$alpha (output directory exists and is non-empty: $output_dir)"
        echo "=========================================="
        echo ""
        continue
    fi

    echo "=========================================="
    echo "Running experiment with alpha=$alpha"
    echo "Experiment name: adversarial_sft_alpha_${alpha_str}"
    echo "Output directory: $output_dir"
    echo "=========================================="

    uv run python -m exp.adversarial_sft \
        --config-name config_adversarial \
        training=adversarial \
        training.adversarial.alpha="$alpha" \
        experiment.name=adversarial_sft_alpha_${alpha_str} \
        experiment.output_dir="$output_dir" \
        "${HYDRA_ARGS[@]}"

    echo ""
    echo "Completed experiment with alpha=$alpha"
    echo ""
done

echo "=========================================="
echo "Alpha sweep completed!"
echo "=========================================="
