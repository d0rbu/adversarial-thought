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
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import hydra
import wandb
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
    assert hasattr(cfg, "questions"), "Config must have 'questions' section"
    assert hasattr(cfg.questions, "split"), "Config must have 'questions.split'"
    assert hasattr(
        cfg.questions, "n_questions"
    ), "Config must have 'questions.n_questions'"

    split = assert_type(cfg.questions.split, str)
    n_questions = cfg.questions.n_questions

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
    if n_questions is not None:
        n_questions = assert_type(n_questions, int)
        assert n_questions > 0, f"n_questions must be positive, got {n_questions}"
        if n_questions < len(questions):
            questions = questions[:n_questions]

    assert len(questions) > 0, "Must have at least one question"
    assert all(isinstance(q, str) for q in questions), "All questions must be strings"
    assert all(len(q) > 0 for q in questions), "All questions must be non-empty"
    return questions


def get_context_prompts(cfg: DictConfig) -> list[str]:
    """Get context prompts based on configuration.

    If source is "dataset", loads the validation set using the same split
    logic as training (same seed, train_ratio, max_samples) to ensure
    we're evaluating on the same validation data.
    """
    assert hasattr(cfg, "context"), "Config must have 'context' section"
    assert hasattr(cfg.context, "source"), "Config must have 'context.source'"

    source = assert_type(cfg.context.source, str)
    if source == "custom":
        assert hasattr(
            cfg.context, "custom_prompts"
        ), "Config must have 'context.custom_prompts' for custom source"
        prompts = list(cfg.context.custom_prompts)
        assert len(prompts) > 0, "custom_prompts cannot be empty"
        assert all(isinstance(p, str) for p in prompts), "All prompts must be strings"
        assert all(len(p) > 0 for p in prompts), "All prompts must be non-empty"

        return prompts

    assert hasattr(cfg, "oracle"), "Config must have 'oracle' section"
    assert hasattr(cfg.oracle, "model_name"), "Config must have 'oracle.model_name'"
    assert hasattr(cfg, "data"), "Config must have 'data' section"
    assert hasattr(cfg.data, "name"), "Config must have 'data.name'"
    assert hasattr(cfg, "experiment"), "Config must have 'experiment' section"
    assert hasattr(cfg.experiment, "seed"), "Config must have 'experiment.seed'"

    # Load tokenizer to format conversations
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.oracle.model_name,
        trust_remote_code=True,
    )
    assert tokenizer is not None, "Failed to load tokenizer"
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None, "EOS token must be set"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load raw dataset and apply same split logic as training
    logger.info(f"Loading dataset: {cfg.data.name}")
    raw_dataset = hf_load_dataset(cfg.data.name, split="train")
    # Assert we got a Dataset (not DatasetDict or IterableDataset)
    raw_dataset = assert_type(raw_dataset, Dataset)

    assert len(raw_dataset) > 0, "Dataset must not be empty"

    # Apply same filtering and splitting as training
    if cfg.data.max_samples is not None:
        max_samples = assert_type(cfg.data.max_samples, int)
        logger.info(f"Limiting dataset to {max_samples} samples")
        assert max_samples > 0, f"max_samples must be positive, got {max_samples}"
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        assert len(raw_dataset) > 0, "Dataset must not be empty after filtering"

    seed = assert_type(cfg.experiment.seed, int)
    assert isinstance(seed, int), "Seed must be an integer"
    raw_dataset = raw_dataset.shuffle(seed=seed)

    assert hasattr(cfg.data, "split"), "Config must have 'data.split' section"
    assert hasattr(
        cfg.data.split, "train_ratio"
    ), "Config must have 'data.split.train_ratio'"
    train_ratio = assert_type(cfg.data.split.train_ratio, float)
    assert (
        0.0 < train_ratio < 1.0
    ), f"train_ratio must be between 0 and 1, got {train_ratio}"

    split_result = raw_dataset.train_test_split(
        test_size=1 - train_ratio,
        seed=seed,
    )
    assert "test" in split_result, "Split result must have 'test' key"
    val_dataset = split_result["test"]
    assert isinstance(val_dataset, Dataset), "Validation dataset must be a Dataset"
    assert len(val_dataset) > 0, "Validation dataset must not be empty"

    logger.info(
        f"Split dataset: {len(split_result['train'])} train, "
        f"{len(val_dataset)} validation"
    )

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
    if cfg.context.n_samples is not None:
        n_samples = assert_type(cfg.context.n_samples, int)
        assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
        if n_samples < len(contexts):
            contexts = contexts[:n_samples]

    assert len(contexts) > 0, "Must have at least one context prompt"
    assert all(isinstance(c, str) for c in contexts), "All contexts must be strings"
    assert all(len(c) > 0 for c in contexts), "All contexts must be non-empty"
    logger.info(f"Extracted {len(contexts)} context prompts from validation set")
    return contexts


def config_to_oracle_config(cfg: DictConfig) -> OracleConfig:
    """Convert Hydra config to OracleConfig."""
    assert hasattr(cfg, "oracle"), "Config must have 'oracle' section"
    assert hasattr(cfg, "hardware"), "Config must have 'hardware' section"
    assert hasattr(cfg, "judge"), "Config must have 'judge' section"

    assert hasattr(cfg.oracle, "model_name"), "Config must have 'oracle.model_name'"
    assert hasattr(cfg.oracle, "oracle_path"), "Config must have 'oracle.oracle_path'"
    assert hasattr(
        cfg.oracle, "target_adapter_path"
    ), "Config must have 'oracle.target_adapter_path'"
    assert hasattr(
        cfg.oracle, "layer_percent"
    ), "Config must have 'oracle.layer_percent'"
    assert hasattr(
        cfg.oracle, "max_new_tokens"
    ), "Config must have 'oracle.max_new_tokens'"
    assert hasattr(cfg.oracle, "temperature"), "Config must have 'oracle.temperature'"
    assert hasattr(cfg.oracle, "do_sample"), "Config must have 'oracle.do_sample'"
    assert hasattr(cfg.oracle, "batch_size"), "Config must have 'oracle.batch_size'"
    assert hasattr(cfg.hardware, "device"), "Config must have 'hardware.device'"
    assert hasattr(cfg.hardware, "dtype"), "Config must have 'hardware.dtype'"
    assert hasattr(cfg.judge, "model"), "Config must have 'judge.model'"
    assert hasattr(cfg.judge, "max_tokens"), "Config must have 'judge.max_tokens'"

    model_name = assert_type(cfg.oracle.model_name, str)
    oracle_path = assert_type(cfg.oracle.oracle_path, str)
    layer_percent = assert_type(cfg.oracle.layer_percent, int)
    max_new_tokens = assert_type(cfg.oracle.max_new_tokens, int)
    temperature = assert_type(cfg.oracle.temperature, float)
    do_sample = assert_type(cfg.oracle.do_sample, bool)
    batch_size = assert_type(cfg.oracle.batch_size, int)
    device = assert_type(cfg.hardware.device, str)
    dtype = assert_type(cfg.hardware.dtype, str)
    judge_model = assert_type(cfg.judge.model, str)
    judge_max_tokens = assert_type(cfg.judge.max_tokens, int)

    assert len(model_name) > 0, "model_name cannot be empty"
    assert len(oracle_path) > 0, "oracle_path cannot be empty"
    assert (
        0 < layer_percent <= 100
    ), f"layer_percent must be between 0 and 100, got {layer_percent}"
    assert max_new_tokens > 0, f"max_new_tokens must be positive, got {max_new_tokens}"
    assert temperature >= 0.0, f"temperature must be non-negative, got {temperature}"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert len(device) > 0, "device cannot be empty"
    assert len(dtype) > 0, "dtype cannot be empty"
    assert len(judge_model) > 0, "judge_model cannot be empty"
    assert (
        judge_max_tokens > 0
    ), f"judge_max_tokens must be positive, got {judge_max_tokens}"

    return OracleConfig(
        model_name=model_name,
        oracle_path=oracle_path,
        target_adapter_path=cfg.oracle.target_adapter_path,
        layer_percent=layer_percent,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        judge_model=judge_model,
        judge_max_tokens=judge_max_tokens,
    )


def compute_metrics(results: OracleEvalResults) -> dict[str, Any]:
    """Compute summary metrics from oracle results."""
    assert results is not None, "Results cannot be None"
    assert hasattr(results, "results"), "Results must have 'results' attribute"
    assert len(results.results) > 0, "Results cannot be empty"

    # Score distribution - all results should have scores (judge is always enabled)
    scores = [r.judge_score for r in results.results]
    assert len(scores) == len(results.results), "All results must have judge scores"
    assert all(1 <= s <= 5 for s in scores), "All scores must be between 1 and 5"

    # Verify all results have required fields
    for r in results.results:
        assert hasattr(r, "context"), "Result must have 'context' attribute"
        assert hasattr(r, "question"), "Result must have 'question' attribute"
        assert hasattr(r, "judge_score"), "Result must have 'judge_score' attribute"
        assert hasattr(
            r, "oracle_response"
        ), "Result must have 'oracle_response' attribute"
        assert len(r.context) > 0, "Context cannot be empty"
        assert len(r.question) > 0, "Question cannot be empty"
        assert len(r.oracle_response) > 0, "Oracle response cannot be empty"

    metrics: dict[str, Any] = {
        "total_queries": len(results.results),
        "unique_contexts": len({r.context for r in results.results}),
        "unique_questions": len({r.question for r in results.results}),
        "mean_score": results.mean_score(),
    }

    assert metrics["total_queries"] > 0, "total_queries must be positive"
    assert metrics["unique_contexts"] > 0, "unique_contexts must be positive"
    assert metrics["unique_questions"] > 0, "unique_questions must be positive"
    assert (
        1.0 <= metrics["mean_score"] <= 5.0
    ), f"mean_score must be between 1 and 5, got {metrics['mean_score']}"

    # Score distribution
    metrics["score_distribution"] = {str(i): scores.count(i) for i in range(1, 6)}
    metrics["min_score"] = min(scores)
    metrics["max_score"] = max(scores)

    assert metrics["min_score"] >= 1, "min_score must be at least 1"
    assert metrics["max_score"] <= 5, "max_score must be at most 5"
    assert sum(metrics["score_distribution"].values()) == len(
        scores
    ), "Score distribution must sum to number of scores"

    # Response length stats
    response_lengths = [len(r.oracle_response) for r in results.results]
    assert len(response_lengths) > 0, "Response lengths cannot be empty"
    metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
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
    assert results is not None, "Results cannot be None"
    assert metrics is not None, "Metrics cannot be None"
    assert len(output_dir) > 0, "output_dir cannot be empty"
    assert hasattr(cfg, "oracle"), "Config must have 'oracle' section"
    assert hasattr(cfg, "questions"), "Config must have 'questions' section"
    assert hasattr(cfg, "judge"), "Config must have 'judge' section"

    output_path = Path(output_dir)
    assert (
        output_path.parent.exists() or str(output_path.parent) == "."
    ), f"Parent directory must exist: {output_path.parent}"
    output_path.mkdir(parents=True, exist_ok=True)

    yaml_file = output_path / "oracle_results.yaml"
    assert yaml_file.parent.exists(), "Output directory must exist"

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
    assert cfg is not None, "Config cannot be None"
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
        assert hasattr(cfg, "wandb"), "Config must have 'wandb' section"
        assert hasattr(cfg.wandb, "project"), "Config must have 'wandb.project'"
        assert hasattr(cfg, "experiment"), "Config must have 'experiment' section"
        assert hasattr(cfg.experiment, "name"), "Config must have 'experiment.name'"

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        assert config_dict is not None, "Config dict cannot be None"
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
    assert hasattr(
        cfg.experiment, "output_dir"
    ), "Config must have 'experiment.output_dir'"
    output_dir = cfg.experiment.output_dir
    assert isinstance(output_dir, str), "output_dir must be a string"
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
