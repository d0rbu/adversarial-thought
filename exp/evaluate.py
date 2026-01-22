"""Evaluation module using lm-eval-harness for model benchmarking."""

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lm_eval
import numpy as np
import torch as t
import yaml
from lm_eval.models.huggingface import HFLM
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra
import wandb
from core.dtype import get_dtype


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Model settings
    model_name: str = "google/gemma-3-1b-it"
    tokenizer_name: str | None = None  # Defaults to model_name if None

    # PEFT adapter settings (for finetuned models)
    peft_adapter_path: str | None = None  # Path to PEFT adapter, None for base model

    # Evaluation tasks
    tasks: list[str] = field(
        default_factory=lambda: [
            "hellaswag",
            "winogrande",
            "mmlu",
            "hendrycks_math",
            "gsm8k",
            "xquad",
        ]
    )
    num_fewshot: int | None = None  # None uses task defaults

    # Evaluation settings
    batch_size: int | str = "auto"  # "auto" or integer
    max_batch_size: int | None = None  # Max batch size when using "auto"
    limit: int | None = None  # Limit number of examples per task (for testing)

    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Output settings
    output_dir: str = "out/eval"
    seed: int = 42

    # W&B settings
    wandb_enabled: bool = True
    wandb_project: str = "adversarial-thought"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def create_lm_eval_model(cfg: EvalConfig) -> HFLM:
    """Create an lm-eval HFLM model instance.

    Supports both base models and models with PEFT adapters.

    Args:
        cfg: Evaluation configuration

    Returns:
        HFLM model instance ready for evaluation
    """
    logger.info(f"Loading model: {cfg.model_name}")

    torch_dtype = get_dtype(cfg.dtype)
    tokenizer_name = cfg.tokenizer_name or cfg.model_name

    # Common model kwargs
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if cfg.device == "cuda":
        model_kwargs["device_map"] = "auto"

    # If we have a PEFT adapter, load it
    if cfg.peft_adapter_path is not None:
        logger.info(f"Loading PEFT adapter from: {cfg.peft_adapter_path}")
        model = HFLM(
            pretrained=cfg.model_name,
            tokenizer=tokenizer_name,
            peft=cfg.peft_adapter_path,
            dtype=str(torch_dtype).split(".")[-1],  # "bfloat16", "float16", etc.
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            trust_remote_code=True,
            device=cfg.device,
        )
    else:
        logger.info("Loading base model (no PEFT adapter)")
        model = HFLM(
            pretrained=cfg.model_name,
            tokenizer=tokenizer_name,
            dtype=str(torch_dtype).split(".")[-1],
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            trust_remote_code=True,
            device=cfg.device,
        )

    return model


def run_evaluation(
    model: HFLM,
    tasks: list[str],
    num_fewshot: int | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run lm-eval evaluation on the specified tasks.

    Args:
        model: HFLM model instance
        tasks: List of task names to evaluate
        num_fewshot: Number of few-shot examples (None for task defaults)
        limit: Limit number of examples per task (None for all)
        seed: Random seed for evaluation

    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Running evaluation on tasks: {tasks}")

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )

    return results


def extract_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Extract key metrics from lm-eval results.

    Args:
        results: Raw results dictionary from lm-eval

    Returns:
        Flattened dictionary of metric_name -> value
    """
    metrics: dict[str, float] = {}

    if "results" not in results:
        return metrics

    for task_name, task_results in results["results"].items():
        for metric_name, value in task_results.items():
            # Skip stderr and other metadata
            if metric_name.endswith(",none"):
                # Extract the actual metric name (e.g., "acc,none" -> "acc")
                clean_metric = metric_name.replace(",none", "")
                key = f"{task_name}/{clean_metric}"
                if isinstance(value, int | float):
                    metrics[key] = float(value)
            elif metric_name == "acc" or metric_name == "acc_norm":
                key = f"{task_name}/{metric_name}"
                if isinstance(value, int | float):
                    metrics[key] = float(value)

    return metrics


def format_metrics_by_task(metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Organize metrics by task for cleaner output.

    Args:
        metrics: Flat dictionary of "task/metric" -> value

    Returns:
        Nested dictionary of task -> {metric -> value}
    """
    by_task: dict[str, dict[str, float]] = {}

    for key, value in sorted(metrics.items()):
        if "/" in key:
            task, metric = key.rsplit("/", 1)
        else:
            task, metric = "unknown", key

        if task not in by_task:
            by_task[task] = {}
        by_task[task][metric] = round(value, 4)

    return by_task


def save_results_yaml(
    metrics: dict[str, float],
    output_dir: str,
    config: EvalConfig,
) -> Path:
    """Save human-readable YAML summary of evaluation results.

    Args:
        metrics: Extracted metrics dictionary
        output_dir: Directory to save results
        config: Evaluation configuration

    Returns:
        Path to the saved YAML file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine model identifier for filename
    if config.peft_adapter_path:
        model_id = Path(config.peft_adapter_path).name
    else:
        model_id = config.model_name.replace("/", "_")

    yaml_file = output_path / f"eval_{model_id}.yaml"

    # Organize metrics by task for readability
    metrics_by_task = format_metrics_by_task(metrics)

    # Create human-readable summary
    summary = {
        "evaluation_summary": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": config.model_name,
            "adapter": config.peft_adapter_path,
            "tasks": config.tasks,
            "num_fewshot": config.num_fewshot,
            "limit": config.limit,
            "seed": config.seed,
        },
        "results": metrics_by_task,
    }

    with yaml_file.open("w") as f:
        yaml.dump(
            summary, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    logger.info(f"Summary saved to: {yaml_file}")
    return yaml_file


def save_results(
    results: dict[str, Any],
    metrics: dict[str, float],
    output_dir: str,
    config: EvalConfig,
) -> tuple[Path, Path]:
    """Save evaluation results to disk (both full JSON and summary YAML).

    Args:
        results: Full results dictionary from lm-eval
        metrics: Extracted metrics dictionary
        output_dir: Directory to save results
        config: Evaluation configuration

    Returns:
        Tuple of (json_path, yaml_path) for the saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine model identifier for filename
    if config.peft_adapter_path:
        model_id = Path(config.peft_adapter_path).name
    else:
        model_id = config.model_name.replace("/", "_")

    # Save full JSON results
    json_file = output_path / f"eval_{model_id}.json"

    output_data = {
        "config": {
            "model_name": config.model_name,
            "peft_adapter_path": config.peft_adapter_path,
            "tasks": config.tasks,
            "num_fewshot": config.num_fewshot,
            "limit": config.limit,
            "seed": config.seed,
        },
        "metrics": metrics,
        "full_results": results.get("results", {}),
    }

    with json_file.open("w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Full results saved to: {json_file}")

    # Save human-readable YAML summary
    yaml_file = save_results_yaml(metrics, output_dir, config)

    return json_file, yaml_file


def config_to_eval_config(cfg: DictConfig) -> EvalConfig:
    """Convert Hydra DictConfig to EvalConfig dataclass."""
    return EvalConfig(
        model_name=cfg.model.name,
        tokenizer_name=cfg.model.tokenizer if hasattr(cfg.model, "tokenizer") else None,
        peft_adapter_path=cfg.eval.peft_adapter_path
        if hasattr(cfg.eval, "peft_adapter_path")
        else None,
        tasks=list(cfg.eval.tasks),
        num_fewshot=cfg.eval.num_fewshot if hasattr(cfg.eval, "num_fewshot") else None,
        batch_size=cfg.eval.batch_size,
        max_batch_size=cfg.eval.max_batch_size
        if hasattr(cfg.eval, "max_batch_size")
        else None,
        limit=cfg.eval.limit if hasattr(cfg.eval, "limit") else None,
        device=cfg.hardware.device,
        dtype=cfg.hardware.dtype,
        output_dir=cfg.experiment.output_dir,
        seed=cfg.experiment.seed,
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    logger.info("Starting model evaluation")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Convert to typed config
    eval_cfg = config_to_eval_config(cfg)

    # Set seed for reproducibility
    set_seed(eval_cfg.seed)

    # Initialize W&B if enabled
    if eval_cfg.wandb_enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        run_name = f"eval_{cfg.experiment.name}"
        if eval_cfg.peft_adapter_path:
            run_name += f"_{Path(eval_cfg.peft_adapter_path).name}"

        wandb.init(
            project=eval_cfg.wandb_project,
            name=run_name,
            config=cast("dict[str, Any]", config_dict)
            if isinstance(config_dict, dict)
            else None,
            tags=["eval"] + (list(cfg.wandb.tags) if cfg.wandb.tags else []),
        )

    # Create model
    model = create_lm_eval_model(eval_cfg)

    # Run evaluation
    results = run_evaluation(
        model=model,
        tasks=eval_cfg.tasks,
        num_fewshot=eval_cfg.num_fewshot,
        limit=eval_cfg.limit,
        seed=eval_cfg.seed,
    )

    # Extract and log metrics
    metrics = extract_metrics(results)
    logger.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2)}")

    # Log to W&B
    if eval_cfg.wandb_enabled:
        wandb.log(metrics)

        # Create a summary table
        table = wandb.Table(columns=["Task", "Metric", "Value"])
        for key, value in metrics.items():
            task, metric = key.rsplit("/", 1)
            table.add_data(task, metric, value)
        wandb.log({"eval_results": table})

    # Save results
    save_results(results, metrics, eval_cfg.output_dir, eval_cfg)

    # Finish W&B run
    if eval_cfg.wandb_enabled:
        wandb.finish()

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
