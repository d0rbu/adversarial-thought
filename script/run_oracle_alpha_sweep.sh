#!/usr/bin/env bash
# Run activation oracle evaluation (validation question split) for all adversarial
# alpha sweep checkpoints.
#
# Produces `out/**/oracle_results.yaml` artifacts consumed by:
#   - viz/plot_pareto_alpha_sweep.py
#
# Usage:
#   ./script/run_oracle_alpha_sweep.sh [--start-from ALPHA] [--no-skip] [hydra overrides...]
#
# Examples:
#   ./script/run_oracle_alpha_sweep.sh
#   ./script/run_oracle_alpha_sweep.sh --start-from 0.004
#   ./script/run_oracle_alpha_sweep.sh questions.n_questions=10
#   ./script/run_oracle_alpha_sweep.sh oracle.context.n_samples=64
#   ./script/run_oracle_alpha_sweep.sh wandb.enabled=false
#   ./script/run_oracle_alpha_sweep.sh judge.max_tokens=500

set -euo pipefail

cd "$(dirname "$0")/.."

# Alpha values to sweep through (must match run_adversarial_sft_alpha_sweep.sh)
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)

# Parse options
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

echo "Running oracle (val split) for alpha sweep: ${ALPHA_VALUES[*]}"
echo "Starting from index: $START_INDEX (alpha=${ALPHA_VALUES[$START_INDEX]})"
echo "Total alphas: ${#ALPHA_VALUES[@]}"
echo "Skip mode: $SKIP_MODE"
echo ""

for ((i=$START_INDEX; i<${#ALPHA_VALUES[@]}; i++)); do
    alpha="${ALPHA_VALUES[$i]}"
    alpha_str=$(echo "$alpha" | tr '.' '_')

    adapter_path="out/adversarial_sft_alpha_${alpha_str}"
    oracle_output_dir="out/oracle_adversarial_sft_alpha_${alpha_str}"
    oracle_yaml="${oracle_output_dir}/oracle_results.yaml"

    if [[ "$SKIP_MODE" == "auto" && -f "$oracle_yaml" && -s "$oracle_yaml" ]]; then
        echo "=========================================="
        echo "Skipping alpha=$alpha (existing oracle YAML: $oracle_yaml)"
        echo "=========================================="
        echo ""
        continue
    fi

    echo "=========================================="
    echo "Running oracle eval for alpha=$alpha"
    echo "Adapter path    : $adapter_path"
    echo "Output dir      : $oracle_output_dir"
    echo "=========================================="

    uv run python -m exp.run_oracle \
        oracle=sft \
        oracle.target_adapter_path="$adapter_path" \
        model.load_in_8bit=false \
        hardware.dtype=bfloat16 \
        experiment.name="oracle_adversarial_sft_alpha_${alpha_str}" \
        experiment.output_dir="$oracle_output_dir" \
        questions.split=val \
        "${HYDRA_ARGS[@]}"

    echo ""
    echo "Completed oracle eval for alpha=$alpha"
    echo ""
done

echo "=========================================="
echo "Oracle alpha sweep completed!"
echo "=========================================="
