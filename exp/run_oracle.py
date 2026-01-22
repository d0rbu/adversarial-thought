"""Run activation oracle evaluation with Hydra configuration.

This script runs the activation oracle on specified prompts and questions,
collecting responses about what information is encoded in model activations
at the last token position. An LLM judge scores each response 1-5.

Setup:
    Set OPENAI_API_KEY environment variable for the LLM judge.

Usage:
    uv run python -m exp.run_oracle
    uv run python -m exp.run_oracle oracle=sft
    uv run python -m exp.run_oracle questions.n_questions=5
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra
import wandb
from core.questions import get_train_questions, get_val_questions
from exp.oracle import (
    OracleConfig,
    OracleEvalResults,
    run_oracle_eval,
    save_oracle_results,
)


def get_questions_for_eval(cfg: DictConfig) -> list[str]:
    """Get questions based on configuration."""
    split = cfg.questions.split  # "train" or "val"
    n_questions = cfg.questions.n_questions

    if split == "train":
        questions = get_train_questions()
    elif split == "val":
        questions = get_val_questions()
    elif split == "all":
        questions = get_train_questions() + get_val_questions()
    else:
        raise ValueError(f"Unknown question split: {split}")

    # Limit number of questions if specified
    if n_questions is not None and n_questions < len(questions):
        questions = questions[:n_questions]

    return questions


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
        max_new_tokens=cfg.oracle.max_new_tokens,
        temperature=cfg.oracle.temperature,
        do_sample=cfg.oracle.do_sample,
        batch_size=cfg.oracle.batch_size,
        device=cfg.hardware.device,
        dtype=cfg.hardware.dtype,
        judge_model=cfg.judge.model,
        judge_max_tokens=cfg.judge.max_tokens,
    )


def compute_metrics(results: OracleEvalResults) -> dict[str, Any]:
    """Compute summary metrics from oracle results."""
    metrics: dict[str, Any] = {
        "total_queries": len(results.results),
        "unique_contexts": len({r.context for r in results.results}),
        "unique_questions": len({r.question for r in results.results}),
        "mean_score": results.mean_score,
    }

    # Score distribution
    scores = [r.judge_score for r in results.results if r.judge_score is not None]
    if scores:
        metrics["score_distribution"] = {str(i): scores.count(i) for i in range(1, 6)}
        metrics["min_score"] = min(scores)
        metrics["max_score"] = max(scores)

    # Response length stats
    response_lengths = [len(r.oracle_response) for r in results.results]
    if response_lengths:
        metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)

    return metrics


def save_results_yaml(
    results: OracleEvalResults,
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
            "question_split": cfg.questions.split,
            "judge_model": cfg.judge.model,
        },
        "metrics": {
            "mean_score": round(metrics["mean_score"], 3)
            if metrics.get("mean_score")
            else None,
            "total_queries": metrics["total_queries"],
            "score_distribution": metrics.get("score_distribution"),
        },
        "sample_results": [
            {
                "context": r.context[:100] + "..."
                if len(r.context) > 100
                else r.context,
                "question": r.question,
                "response": r.oracle_response[:200] + "..."
                if len(r.oracle_response) > 200
                else r.oracle_response,
                "score": r.judge_score,
                "reasoning": r.judge_reasoning,
            }
            for r in results.results[:10]
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

    # Convert config and run evaluation
    oracle_cfg = config_to_oracle_config(cfg)
    results = run_oracle_eval(
        config=oracle_cfg,
        questions=questions,
        contexts=context_prompts,
    )

    # Compute and log metrics
    metrics = compute_metrics(results)
    logger.info(f"Metrics:\n{json.dumps(metrics, indent=2, default=str)}")

    if cfg.wandb.enabled:
        # Log scalar metrics
        wandb.log(
            {
                "mean_score": metrics.get("mean_score"),
                "total_queries": metrics["total_queries"],
                "avg_response_length": metrics.get("avg_response_length"),
            }
        )

        # Log score distribution as histogram
        if "score_distribution" in metrics:
            for score, count in metrics["score_distribution"].items():
                wandb.log({f"score_{score}": count})

        # Log sample results as table
        table = wandb.Table(
            columns=["Context", "Question", "Response", "Score", "Reasoning"]
        )
        for r in results.results[:50]:
            table.add_data(
                r.context[:100],
                r.question,
                r.oracle_response[:200],
                r.judge_score,
                r.judge_reasoning,
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
    if results.mean_score is not None:
        logger.info(f"Final mean score: {results.mean_score:.2f}/5.0")


if __name__ == "__main__":
    main()
