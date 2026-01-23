"""Evaluation module using lm-eval-harness for model benchmarking."""

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lm_eval
import numpy as np
import torch as t
import yaml
from lm_eval.models.huggingface import HFLM
from lm_eval.models.huggingface import get_dtype as hflm_get_dtype
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import hydra
import wandb
from core.dtype import get_dtype
from core.type import assert_type


def _clear_task_cache(tasks: list[str]) -> None:
    """Clear HuggingFace dataset cache for evaluation tasks."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    if not cache_dir.exists():
        return

    all_cache_dirs = []
    for task in tasks:
        # Common patterns for task dataset names
        patterns = [
            f"*{task}*",
            f"*{task.replace('_', '___')}*",
            f"*{task.replace('_', '-')}*",
        ]
        if task == "hellaswag":
            patterns.extend(["*Rowan*", "*hellaswag*", "*HellaSwag*"])
        for pattern in patterns:
            all_cache_dirs.extend(cache_dir.glob(pattern))

    # Check subdirectories
    for subdir in cache_dir.iterdir():
        if subdir.is_dir() and any(
            task.lower() in subdir.name.lower() for task in tasks
        ):
            all_cache_dirs.append(subdir)

    # Remove duplicates and clear
    for cache_path in set(all_cache_dirs):
        if cache_path.is_dir():
            logger.info(f"Removing cache directory: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)


class Gemma3CompatibleHFLM(HFLM):
    """HFLM subclass that fixes dtype handling for Gemma3 models.

    Gemma3 models don't accept 'dtype' parameter in their __init__(), but HFLM
    passes it. This subclass overrides _create_model to use 'torch_dtype' instead.
    """

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | t.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config: Any = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """Override _create_model to fix dtype handling for Gemma3 models."""
        model_kwargs = kwargs or {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        # Convert dtype to torch_dtype for Gemma3 compatibility
        # If torch_dtype is already in kwargs, use it; otherwise convert dtype
        if "torch_dtype" in model_kwargs:
            torch_dtype = model_kwargs["torch_dtype"]
        elif dtype is not None:
            torch_dtype = hflm_get_dtype(dtype)
        else:
            torch_dtype = None

        if not autogptq and not gptqmodel:
            if model_kwargs.get("load_in_4bit"):
                import transformers
                from packaging import version as vparse

                assert vparse.parse(transformers.__version__) >= vparse.parse(
                    "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
                if compute_dtype := model_kwargs.get("bnb_4bit_compute_dtype"):
                    model_kwargs["bnb_4bit_compute_dtype"] = hflm_get_dtype(
                        compute_dtype
                    )

            # Use torch_dtype instead of dtype for Gemma3 compatibility
            model_kwargs.pop(
                "torch_dtype", None
            )  # Remove from kwargs to avoid duplication
            if self.AUTO_MODEL_CLASS is not None:
                self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                    pretrained,
                    revision=revision,
                    torch_dtype=torch_dtype,  # Use torch_dtype instead of dtype
                    trust_remote_code=trust_remote_code,
                    gguf_file=gguf_file,
                    quantization_config=quantization_config,
                    subfolder=subfolder,
                    **model_kwargs,
                )
            else:
                raise RuntimeError("AUTO_MODEL_CLASS is not set")
        else:
            # For GPTQ models, use parent implementation
            return super()._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                quantization_config=quantization_config,
                subfolder=subfolder,
                **kwargs,
            )

        # Handle PEFT loading (same as parent)
        if peft:
            import logging

            from peft import PeftModel

            eval_logger = logging.getLogger(__name__)

            if self.tokenizer is not None:
                vocab_size = len(self.tokenizer)
                if (
                    hasattr(self._model.config, "vocab_size")
                    and self._model.config.vocab_size != vocab_size
                ):
                    eval_logger.info(
                        f"Model config indicates vocab_size='{self._model.config.vocab_size}', but found tokenizer with vocab size '{vocab_size}'. Resizing model embedding layer..."
                    )
                    self._model.resize_token_embeddings(vocab_size)
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )


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


def create_lm_eval_model(cfg: EvalConfig) -> Gemma3CompatibleHFLM:
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

    # Create model with optional PEFT adapter
    if cfg.peft_adapter_path is not None:
        logger.info(f"Loading PEFT adapter from: {cfg.peft_adapter_path}")
        model = Gemma3CompatibleHFLM(
            pretrained=cfg.model_name,
            tokenizer=tokenizer_name,
            peft=cfg.peft_adapter_path,
            dtype=None,  # Set to None, torch_dtype will be used instead
            torch_dtype=torch_dtype,  # Pass torch_dtype via kwargs for Gemma3 compatibility
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            trust_remote_code=True,
            device=cfg.device,
        )
    else:
        logger.info("Loading base model (no PEFT adapter)")
        model = Gemma3CompatibleHFLM(
            pretrained=cfg.model_name,
            tokenizer=tokenizer_name,
            dtype=None,  # Set to None, torch_dtype will be used instead
            torch_dtype=torch_dtype,  # Pass torch_dtype via kwargs for Gemma3 compatibility
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
    assert len(tasks) > 0, "Must have at least one task"
    assert all(len(t) > 0 for t in tasks), "All task names must be non-empty"
    if num_fewshot is not None:
        assert num_fewshot >= 0, f"num_fewshot must be non-negative, got {num_fewshot}"
    if limit is not None:
        assert limit > 0, f"limit must be positive, got {limit}"

    logger.info(f"Running evaluation on tasks: {tasks}")

    try:
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
        )
    except ValueError as e:
        if "Feature type 'List' not found" in str(e):
            logger.warning(
                "Dataset cache has incompatible format. Clearing cache and retrying..."
            )
            _clear_task_cache(tasks)
            results = lm_eval.simple_evaluate(
                model=model,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
            )
        else:
            raise

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
    assert metrics is not None and isinstance(
        metrics, dict
    ), "Metrics must be a dictionary"
    assert len(output_dir) > 0, "output_dir cannot be empty"
    assert config is not None, "Config cannot be None"

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
    assert results is not None and isinstance(
        results, dict
    ), "Results must be a dictionary"
    assert metrics is not None and isinstance(
        metrics, dict
    ), "Metrics must be a dictionary"
    assert len(output_dir) > 0, "output_dir cannot be empty"
    assert config is not None, "Config cannot be None"

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
    model_name = assert_type(cfg.model.name, str)
    tasks = list(cfg.eval.tasks)
    device = assert_type(cfg.hardware.device, str)
    dtype = assert_type(cfg.hardware.dtype, str)
    output_dir = assert_type(cfg.experiment.output_dir, str)
    seed = assert_type(cfg.experiment.seed, int)
    wandb_enabled = assert_type(cfg.wandb.enabled, bool)
    wandb_project = assert_type(cfg.wandb.project, str)

    assert len(tasks) > 0, "tasks cannot be empty"

    return EvalConfig(
        model_name=model_name,
        tokenizer_name=cfg.model.tokenizer if hasattr(cfg.model, "tokenizer") else None,
        peft_adapter_path=cfg.eval.peft_adapter_path
        if hasattr(cfg.eval, "peft_adapter_path")
        else None,
        tasks=tasks,
        num_fewshot=cfg.eval.num_fewshot if hasattr(cfg.eval, "num_fewshot") else None,
        batch_size=cfg.eval.batch_size,
        max_batch_size=cfg.eval.max_batch_size
        if hasattr(cfg.eval, "max_batch_size")
        else None,
        limit=cfg.eval.limit if hasattr(cfg.eval, "limit") else None,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
        seed=seed,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
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
