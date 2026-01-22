"""Run activation oracle evaluation with Hydra configuration.

This script runs the activation oracle on specified prompts and questions,
collecting responses about what information is encoded in model activations.

Usage (from activation_oracles environment):
    python -m exp.run_oracle
    python -m exp.run_oracle oracle=sft
    python -m exp.run_oracle questions.n_questions=5
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra
import wandb
from core.questions import (
    get_adversarial_questions,
    get_all_questions,
    get_standard_questions,
)
from exp.oracle import OracleConfig, OracleResult, run_oracle_eval, save_oracle_results


def get_questions_for_eval(cfg: DictConfig) -> list[str]:
    """Get questions based on configuration."""
    question_type = cfg.questions.question_type
    n_questions = cfg.questions.n_questions

    if question_type == "all":
        qs = get_all_questions()
    elif question_type == "standard":
        qs = get_standard_questions()
    elif question_type == "adversarial":
        qs = get_adversarial_questions()
    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    # Sample if needed
    if n_questions is not None:
        sampled = qs.sample_train(n=n_questions, seed=cfg.experiment.seed)
    else:
        sampled = qs.train

    return [q.text for q in sampled]


def get_context_prompts(cfg: DictConfig) -> list[str]:
    """Get context prompts based on configuration."""
    if cfg.context.source == "custom":
        return list(cfg.context.custom_prompts)
    elif cfg.context.source == "dataset":
        # TODO: Load from dataset
        raise NotImplementedError("Dataset source not yet implemented")
    else:
        raise ValueError(f"Unknown context source: {cfg.context.source}")


def config_to_oracle_config(cfg: DictConfig) -> OracleConfig:
    """Convert Hydra config to OracleConfig."""
    return OracleConfig(
        model_name=cfg.oracle.model_name,
        oracle_path=cfg.oracle.oracle_path,
        target_adapter_path=cfg.oracle.target_adapter_path,
        layer_percent=cfg.oracle.layer_percent,
        segment_start=cfg.oracle.segment_start,
        max_new_tokens=cfg.oracle.max_new_tokens,
        temperature=cfg.oracle.temperature,
        do_sample=cfg.oracle.do_sample,
        batch_size=cfg.oracle.batch_size,
        device=cfg.hardware.device,
        dtype=cfg.hardware.dtype,
    )


def compute_metrics(results: list[OracleResult]) -> dict[str, Any]:
    """Compute summary metrics from oracle results."""
    metrics: dict[str, Any] = {
        "total_queries": len(results),
        "unique_contexts": len({r.context_prompt for r in results}),
        "unique_questions": len({r.oracle_question for r in results}),
    }

    # Count by activation type
    act_types = {}
    for r in results:
        act_types[r.activation_type] = act_types.get(r.activation_type, 0) + 1
    metrics["activation_types"] = act_types

    # Response length stats
    response_lengths = [len(r.oracle_response) for r in results]
    if response_lengths:
        metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
        metrics["max_response_length"] = max(response_lengths)

    return metrics


def save_results_yaml(
    results: list[OracleResult],
    metrics: dict[str, Any],
    output_dir: str,
    cfg: DictConfig,
) -> Path:
    """Save human-readable YAML summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    yaml_file = output_path / "oracle_results.yaml"

    summary = {
        "oracle_evaluation_summary": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": cfg.oracle.model_name,
            "oracle": cfg.oracle.oracle_path,
            "target_adapter": cfg.oracle.target_adapter_path,
            "n_questions": cfg.questions.n_questions,
            "question_type": cfg.questions.question_type,
        },
        "metrics": metrics,
        "sample_results": [
            {
                "context": r.context_prompt[:100] + "..."
                if len(r.context_prompt) > 100
                else r.context_prompt,
                "question": r.oracle_question,
                "response": r.oracle_response,
            }
            for r in results[:10]  # First 10 results
        ],
    }

    with yaml_file.open("w") as f:
        yaml.dump(
            summary, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    logger.info(f"Summary saved to: {yaml_file}")
    return yaml_file


@hydra.main(version_base=None, config_path="../conf", config_name="oracle_config")
def main(cfg: DictConfig) -> None:
    """Main oracle evaluation function."""
    logger.info("Starting activation oracle evaluation")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get questions and prompts
    questions = get_questions_for_eval(cfg)
    context_prompts = get_context_prompts(cfg)

    logger.info(
        f"Using {len(questions)} questions and {len(context_prompts)} context prompts"
    )

    # Initialize W&B if enabled
    if cfg.wandb.enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.experiment.name,
            config=cast("dict[str, Any]", config_dict)
            if isinstance(config_dict, dict)
            else None,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        )

    # Convert config
    oracle_cfg = config_to_oracle_config(cfg)

    # Run oracle evaluation
    results = run_oracle_eval(
        config=oracle_cfg,
        oracle_questions=questions,
        context_prompts=context_prompts,
    )

    # Compute and log metrics
    metrics = compute_metrics(results)
    logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    if cfg.wandb.enabled:
        wandb.log(metrics)

        # Log sample results as table
        table = wandb.Table(columns=["Context", "Question", "Response"])
        for r in results[:50]:  # First 50
            table.add_data(
                r.context_prompt[:100],
                r.oracle_question,
                r.oracle_response,
            )
        wandb.log({"oracle_results": table})

    # Save results
    output_dir = cfg.experiment.output_dir
    save_oracle_results(results, Path(output_dir) / "oracle_results.json")
    save_results_yaml(results, metrics, output_dir, cfg)

    # Finish W&B
    if cfg.wandb.enabled:
        wandb.finish()

    logger.info("Oracle evaluation complete!")


if __name__ == "__main__":
    main()
