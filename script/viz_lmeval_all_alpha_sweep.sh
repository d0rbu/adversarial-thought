#!/usr/bin/env bash
# Visualize baseline, SFT, and all adversarial alpha-sweep lm-eval results together.
# Expects out/eval_baseline*, out/eval_sft*, and out/eval_adversarial_sft_alpha_* from
# run_eval_baseline.sh, run_eval_sft.sh, and run_eval_alpha_sweep.sh.
#
# Usage: ./script/viz_lmeval_all_alpha_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

shopt -s nullglob

# Alpha values (must match run_eval_alpha_sweep.sh)
ALPHA_VALUES=(0.001 0.004 0.01 0.04 0.1)

BASELINE_YAMLS=(out/eval_baseline/*.yaml out/eval_baseline_quick/*.yaml)
SFT_YAMLS=(out/eval_sft/*.yaml out/eval_sft_quick/*.yaml)

if [[ ${#BASELINE_YAMLS[@]} -eq 0 ]]; then
    echo "No baseline lm-eval YAML found in out/eval_baseline*/" >&2
    echo "Run: ./script/run_eval_baseline.sh" >&2
    exit 1
fi
if [[ ${#SFT_YAMLS[@]} -eq 0 ]]; then
    echo "No SFT lm-eval YAML found in out/eval_sft*/" >&2
    echo "Run: ./script/run_eval_sft.sh" >&2
    exit 1
fi

YAML_PATHS=("${BASELINE_YAMLS[0]}" "${SFT_YAMLS[0]}")
LABELS=("Baseline" "SFT")

for alpha in "${ALPHA_VALUES[@]}"; do
    alpha_str=$(echo "$alpha" | tr '.' '_')
    eval_dir="out/eval_adversarial_sft_alpha_${alpha_str}"
    eval_yaml="${eval_dir}/eval_adversarial_sft_alpha_${alpha_str}.yaml"
    if [[ -f "$eval_yaml" ]]; then
        YAML_PATHS+=("$eval_yaml")
        LABELS+=("Adv ${alpha}")
    fi
done

if [[ ${#YAML_PATHS[@]} -eq 2 ]]; then
    echo "No adversarial alpha-sweep lm-eval results found in out/eval_adversarial_sft_alpha_*" >&2
    echo "Run: ./script/run_eval_alpha_sweep.sh" >&2
    exit 1
fi

# Build comma-separated labels
IFS=','; LABELS_CSV="${LABELS[*]}"; unset IFS

uv run python -m viz.plot_lmeval_grouped \
    "${YAML_PATHS[@]}" \
    --labels "$LABELS_CSV" \
    --output-dir fig \
    --output-prefix all_alpha_sweep
