#!/usr/bin/env bash
# Run adversarial SFT finetuning experiment
# Usage: ./script/run_adversarial_sft.sh [hydra overrides...]
# Examples:
#   ./script/run_adversarial_sft.sh
#   ./script/run_adversarial_sft.sh training.adversarial.alpha=2.0
#   ./script/run_adversarial_sft.sh training.batch_size=2 training.num_epochs=3
#   ./script/run_adversarial_sft.sh training.adversarial.oracle_path=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
#   ./script/run_adversarial_sft.sh wandb.enabled=false
#   ./script/run_adversarial_sft.sh data.max_samples=1000

set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m exp.adversarial_sft \
    --config-name config_adversarial \
    training=adversarial \
    experiment.name=adversarial_sft \
    experiment.output_dir=out/adversarial_sft \
    "$@"
