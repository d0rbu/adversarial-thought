"""Run activation oracle evaluation with Hydra configuration.

This script runs the activation oracle on specified prompts and questions,
collecting responses about what information is encoded in model activations
at the last token position. An LLM judge scores each response 1-5.

Setup:
    Set OPENAI_API_KEY environment variable for the LLM judge.
    You can set it in your environment or in a .env file in the project root.

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
from transformers import AutoTokenizer

import hydra
import wandb
from core.data import load_and_split_dataset
from core.questions import get_train_questions, get_val_questions
from core.type import assert_type
from exp.oracle import (
    OracleConfig,
    OracleEvalResults,
    run_oracle_eval,
    save_oracle_results,
)


def get_questions_for_eval(cfg: DictConfig) -> list[str]:
    """Get questions based on configuration."""
    split = assert_type(cfg.questions.split, str)
    if split == "train":
        questions = get_train_questions()
    elif split == "val":
        questions = get_val_questions()
    elif split == "all":
        questions = get_train_questions() + get_val_questions()
    else:
        raise ValueError(f"Unknown question split: {split}")

    assert len(questions) > 0, "Question lists must not be empty"

    # Limit number of questions if specified
    if cfg.questions.n_questions is not None:
        n_questions = assert_type(cfg.questions.n_questions, int)
        assert n_questions > 0, f"n_questions must be positive, got {n_questions}"
        questions = questions[:n_questions]

    return questions


def get_context_prompts(cfg: DictConfig) -> list[str]:
    """Get context prompts based on configuration.

    If source is "dataset", loads the validation set using the same split
    logic as training (same seed, train_ratio, max_samples) to ensure
    we're evaluating on the same validation data.
    """
    source = assert_type(cfg.oracle.context.source, str)
    if source == "custom":
        prompts = list(cfg.oracle.context.custom_prompts)
        return prompts

    # Load tokenizer to format conversations
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.oracle.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None, "EOS token must be set"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and split dataset using shared helper (same logic as training)
    seed = assert_type(cfg.experiment.seed, int)
    train_ratio = assert_type(cfg.data.split.train_ratio, float)
    max_samples = (
        assert_type(cfg.data.max_samples, int)
        if cfg.data.max_samples is not None
        else None
    )

    datasets = load_and_split_dataset(
        dataset_name=cfg.data.name,
        seed=seed,
        train_ratio=train_ratio,
        max_samples=max_samples,
    )

    val_dataset = datasets["validation"]

    # Extract text from conversations (same cleaning logic as data.py)
    contexts: list[str] = []
    role_map = {"environment": "user"}

    for example in val_dataset:
        messages = example.get("messages", [])
        if not messages:
            continue

        # Clean messages (same logic as in core/data.py)
        cleaned_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            role = role_map.get(role, role)
            content = msg.get("content", "")
            if not content:
                continue
            # Merge consecutive messages with the same role
            if cleaned_messages and cleaned_messages[-1]["role"] == role:
                cleaned_messages[-1]["content"] += "\n\n" + content
            else:
                cleaned_messages.append({"role": role, "content": content})

        # Remove leading assistant messages
        while cleaned_messages and cleaned_messages[0]["role"] == "assistant":
            cleaned_messages.pop(0)

        if not cleaned_messages:
            continue

        # Format as text using chat template
        text = tokenizer.apply_chat_template(
            cleaned_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        contexts.append(str(text))

    # Limit number of contexts if specified
    if (
        hasattr(cfg.oracle.context, "n_samples")
        and cfg.oracle.context.n_samples is not None
    ):
        original_count = len(contexts)
        n_samples = assert_type(cfg.oracle.context.n_samples, int)
        assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
        logger.info(
            f"Config specifies n_samples={n_samples}, limiting from {original_count} contexts"
        )
        if n_samples < len(contexts):
            contexts = contexts[:n_samples]
            logger.info(f"Limited to {len(contexts)} contexts")
        else:
            logger.info(
                f"n_samples={n_samples} >= {original_count} contexts, using all"
            )

    assert len(contexts) > 0, "Must have at least one context prompt"
    assert all(isinstance(c, str) for c in contexts), "All contexts must be strings"
    assert all(
        isinstance(c, str) and len(c) > 0 for c in contexts
    ), "All contexts must be non-empty"
    logger.info(f"Extracted {len(contexts)} context prompts from validation set")
    return contexts


def config_to_oracle_config(cfg: DictConfig) -> OracleConfig:
    """Convert Hydra config to OracleConfig."""
    oracle_cfg = cfg.oracle
    assert all(
        hasattr(oracle_cfg, attr)
        for attr in [
            "model_name",
            "oracle_path",
            "target_adapter_path",
            "layer_percent",
            "max_new_tokens",
            "temperature",
            "do_sample",
            "batch_size",
        ]
    ), "Oracle config missing required fields"
    assert all(
        hasattr(cfg.hardware, attr) for attr in ["device", "dtype"]
    ), "Hardware config missing required fields"
    assert all(
        hasattr(cfg.judge, attr) for attr in ["model", "max_tokens", "temperature"]
    ), "Judge config missing required fields"

    model_name = assert_type(oracle_cfg.model_name, str)
    oracle_path = assert_type(oracle_cfg.oracle_path, str)
    layer_percent = assert_type(oracle_cfg.layer_percent, int)
    max_new_tokens = assert_type(oracle_cfg.max_new_tokens, int)
    temperature = assert_type(oracle_cfg.temperature, float)
    do_sample = assert_type(oracle_cfg.do_sample, bool)
    batch_size = assert_type(oracle_cfg.batch_size, int)
    device = assert_type(cfg.hardware.device, str)
    dtype = assert_type(cfg.hardware.dtype, str)
    judge_model = assert_type(cfg.judge.model, str)
    judge_max_tokens = assert_type(cfg.judge.max_tokens, int)
    judge_temperature = assert_type(cfg.judge.temperature, float)

    assert (
        len(model_name) > 0 and len(oracle_path) > 0
    ), "model_name and oracle_path cannot be empty"
    assert (
        0 < layer_percent <= 100
    ), f"layer_percent must be between 0 and 100, got {layer_percent}"
    assert max_new_tokens > 0, f"max_new_tokens must be positive, got {max_new_tokens}"
    assert temperature >= 0.0, f"temperature must be non-negative, got {temperature}"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert len(device) > 0 and len(dtype) > 0, "device and dtype cannot be empty"
    assert len(judge_model) > 0, "judge_model cannot be empty"
    assert (
        judge_max_tokens > 0
    ), f"judge_max_tokens must be positive, got {judge_max_tokens}"
    assert (
        judge_temperature >= 0.0
    ), f"judge_temperature must be non-negative, got {judge_temperature}"

    return OracleConfig(
        model_name=model_name,
        oracle_path=oracle_path,
        target_adapter_path=oracle_cfg.target_adapter_path,
        layer_percent=layer_percent,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        judge_model=judge_model,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=judge_temperature,
    )


def compute_metrics(results: OracleEvalResults) -> dict[str, Any]:
    """Compute summary metrics from oracle results."""
    assert results is not None and hasattr(
        results, "results"
    ), "Results must have 'results' attribute"
    assert len(results.results) > 0, "Results cannot be empty"

    # Score distribution - all results should have scores (judge is always enabled)
    scores = [r.judge_score for r in results.results]
    assert len(scores) == len(results.results) and all(
        1 <= s <= 5 for s in scores
    ), "All results must have valid judge scores (1-5)"

    # Verify all results have required fields
    for r in results.results:
        assert all(
            hasattr(r, attr)
            for attr in ["context", "question", "judge_score", "oracle_response"]
        ), "Result missing required attributes"
        assert (
            len(r.context) > 0 and len(r.question) > 0 and len(r.oracle_response) > 0
        ), "Context, question, and oracle_response cannot be empty"

    metrics: dict[str, Any] = {
        "total_queries": len(results.results),
        "unique_contexts": len({r.context for r in results.results}),
        "unique_questions": len({r.question for r in results.results}),
        "mean_score": results.mean_score(),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "min_score": min(scores),
        "max_score": max(scores),
        "avg_response_length": sum(len(r.oracle_response) for r in results.results)
        / len(results.results),
    }

    assert metrics["total_queries"] > 0, "total_queries must be positive"
    assert metrics["unique_contexts"] > 0, "unique_contexts must be positive"
    assert metrics["unique_questions"] > 0, "unique_questions must be positive"
    assert (
        1.0 <= metrics["mean_score"] <= 5.0
    ), f"mean_score must be between 1 and 5, got {metrics['mean_score']}"

    # Score distribution
    assert metrics["min_score"] >= 1, "min_score must be at least 1"
    assert metrics["max_score"] <= 5, "max_score must be at most 5"
    assert sum(metrics["score_distribution"].values()) == len(
        scores
    ), "Score distribution must sum to number of scores"

    # Response length stats
    assert (
        metrics["avg_response_length"] >= 0
    ), "avg_response_length must be non-negative"

    return metrics


def save_results_yaml(
    results: OracleEvalResults,
    metrics: dict[str, Any],
    output_dir: str,
    cfg: DictConfig,
) -> Path:
    """Save human-readable YAML summary."""
    assert (
        results is not None and metrics is not None
    ), "Results and metrics cannot be None"
    assert len(output_dir) > 0, "output_dir cannot be empty"
    assert all(
        hasattr(cfg, section) for section in ["oracle", "questions", "judge"]
    ), "Config missing required sections"

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

    assert len(questions) > 0, "Must have at least one question"
    assert len(context_prompts) > 0, "Must have at least one context prompt"
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
    output_dir = assert_type(cfg.experiment.output_dir, str)
    assert len(output_dir) > 0, "output_dir cannot be empty"

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
