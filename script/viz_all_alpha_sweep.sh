#!/usr/bin/env bash
# Visualize baseline, SFT, and all adversarial alpha-sweep oracle results together.
# Expects out/oracle_baseline, out/oracle_sft, and out/oracle_adversarial_sft_alpha_* from
# run_oracle_baseline.sh, run_oracle_sft.sh, and run_oracle_alpha_sweep.sh.
#
# Usage: ./script/viz_all_alpha_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Alpha values (must match run_oracle_alpha_sweep.sh)
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)

BASELINE_YAML="out/oracle_baseline/oracle_results.yaml"
SFT_YAML="out/oracle_sft/oracle_results.yaml"

if [[ ! -f "$BASELINE_YAML" ]]; then
    echo "Baseline oracle results not found: $BASELINE_YAML" >&2
    echo "Run: ./script/run_oracle_baseline.sh" >&2
    exit 1
fi
if [[ ! -f "$SFT_YAML" ]]; then
    echo "SFT oracle results not found: $SFT_YAML" >&2
    echo "Run: ./script/run_oracle_sft.sh" >&2
    exit 1
fi

YAML_PATHS=("$BASELINE_YAML" "$SFT_YAML")
LABELS=("Baseline" "SFT")

for alpha in "${ALPHA_VALUES[@]}"; do
    alpha_str=$(echo "$alpha" | tr '.' '_')
    oracle_yaml="out/oracle_adversarial_sft_alpha_${alpha_str}/oracle_results.yaml"
    if [[ -f "$oracle_yaml" ]]; then
        YAML_PATHS+=("$oracle_yaml")
        LABELS+=("Adv ${alpha}")
    fi
done

if [[ ${#YAML_PATHS[@]} -eq 2 ]]; then
    echo "No adversarial alpha-sweep oracle results found in out/oracle_adversarial_sft_alpha_*" >&2
    echo "Run: ./script/run_oracle_alpha_sweep.sh" >&2
    exit 1
fi

# Build comma-separated labels
IFS=','; LABELS_CSV="${LABELS[*]}"; unset IFS

uv run python -m viz.plot_score_dist_grouped \
    "${YAML_PATHS[@]}" \
    --labels "$LABELS_CSV" \
    --output fig/all_alpha_sweep.png
