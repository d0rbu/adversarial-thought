"""Activation Oracle integration for probing model activations.

This module provides utilities for using activation oracles to query
what information is encoded in model activations at the last token position.
An LLM judge scores the oracle's response accuracy from 1-5.

Setup:
    Set OPENAI_API_KEY environment variable for the LLM judge.
    You can set it in your environment or in a .env file in the project root.

Usage:
    from exp.oracle import OracleConfig, run_oracle_eval

    config = OracleConfig(
        model_name="google/gemma-3-1b-it",
        oracle_path="adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it",
        target_adapter_path="out/sft_baseline",
    )
    results = run_oracle_eval(config, questions, prompts)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from dotenv import load_dotenv
from loguru import logger
from nl_probes.base_experiment import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    load_lora_adapter,
    load_model,
    load_tokenizer,
    run_verbalizer,
)
from openai import OpenAI
from peft import PeftModel

from core.dtype import get_dtype

# Load environment variables from .env file
load_dotenv()


@dataclass
class OracleConfig:
    """Configuration for activation oracle evaluation."""

    # Base model (must match the oracle's base model)
    model_name: str = "google/gemma-3-1b-it"

    # Oracle LoRA path on HuggingFace
    # Available oracles: https://huggingface.co/collections/adamkarvonen/activation-oracles
    oracle_path: str = "adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it"

    # Target adapter path (our finetuned model) - None for base model
    target_adapter_path: str | None = None

    # Which layer to extract activations from (as % of model depth)
    layer_percent: int = 50

    # Generation settings for oracle
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False

    # Batch size for evaluation
    batch_size: int = 32

    # Device and dtype
    device: str = "cuda"
    dtype: str = "bfloat16"

    # LLM judge settings
    judge_model: str = "gpt-5-nano"
    judge_max_tokens: int = 256
    judge_temperature: float = 1.0


@dataclass(frozen=True)
class OracleResult:
    """Result from an oracle query with LLM judge score."""

    context: str
    question: str
    oracle_response: str
    judge_score: int  # 1-5 accuracy score from LLM judge
    judge_reasoning: str

    def __post_init__(self) -> None:
        assert (
            self.judge_score > 0 and self.judge_score <= 5
        ), "Judge score must be between 1 and 5"
        assert len(self.judge_reasoning) > 0, "Judge reasoning cannot be empty"
        assert len(self.oracle_response) > 0, "Oracle response cannot be empty"
        assert len(self.context) > 0, "Context cannot be empty"
        assert len(self.question) > 0, "Question cannot be empty"


@dataclass
class OracleEvalResults:
    """Aggregated results from oracle evaluation."""

    config: OracleConfig
    results: list[OracleResult] = field(default_factory=list)

    def mean_score(self) -> float:
        """Compute mean judge score from results."""
        scores = [r.judge_score for r in self.results]
        if not scores:
            raise ValueError("No scores to compute mean score")

        return sum(scores) / len(scores)


# =============================================================================
# LLM Judge
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI's response about text activations is accurate.

You will be given:
1. CONTEXT: The original text that was processed
2. QUESTION: A question about the context
3. RESPONSE: An AI's answer to that question based on the text's "activations"

Your task: Score the RESPONSE's accuracy from 1-5 based on how well it answers the QUESTION given the CONTEXT.

Scoring rubric:
5 - Completely accurate: Response directly and correctly answers the question
4 - Mostly accurate: Response is largely correct with minor inaccuracies or missing details
3 - Partially accurate: Response contains some correct information but also errors or is vague
2 - Mostly inaccurate: Response is largely wrong or irrelevant
1 - Completely inaccurate: Response is entirely wrong or nonsensical

Respond in JSON format:
{"reasoning": "<brief explanation>", "score": <1-5>}"""


JUDGE_USER_PROMPT = """
CONTEXT: {context}
QUESTION: {question}
RESPONSE: {response}
"""


def judge_response(
    context: str,
    question: str,
    response: str,
    model: str = "gpt-5-nano",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[int, str]:
    """Use LLM judge to score oracle response accuracy.

    Args:
        context: The original text/prompt
        question: Question asked about the context
        response: Oracle's response to evaluate
        model: OpenAI model to use for judging
        max_tokens: Maximum tokens for judge response

    Returns:
        Tuple of (score 1-5, reasoning)
    """
    # Validate inputs
    assert (
        isinstance(context, str) and len(context) > 0
    ), "Context must be a non-empty string"
    assert (
        isinstance(question, str) and len(question) > 0
    ), "Question must be a non-empty string"
    assert (
        isinstance(response, str) and len(response) > 0
    ), "Response must be a non-empty string"
    assert isinstance(model, str) and len(model) > 0, "Model must be a non-empty string"
    assert (
        isinstance(max_tokens, int) and max_tokens > 0
    ), f"max_tokens must be a positive integer, got {max_tokens}"

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY environment variable not set"

    client = OpenAI(api_key=api_key)
    user_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        question=question,
        response=response,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    assert completion is not None, "Completion cannot be None"
    assert len(completion.choices) > 0, "Completion must have at least one choice"
    assert completion.choices[0].message is not None, "Message cannot be None"

    response_text = completion.choices[0].message.content
    assert response_text, "Response was truncated or empty"

    # Parse JSON response - handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    result = json.loads(response_text.strip())
    assert isinstance(result, dict), "Judge response must be a JSON object"
    assert (
        "score" in result and "reasoning" in result
    ), "Judge response must have 'score' and 'reasoning' fields"

    score_raw = result.get("score", 0)
    assert isinstance(score_raw, int), "Score must be an integer"
    score = int(score_raw)

    reasoning = result["reasoning"]
    assert isinstance(reasoning, str), "Reasoning must be a string"
    assert len(reasoning) > 0, "Reasoning cannot be empty"

    # Validate and clamp score range
    if not 1 <= score <= 5:
        logger.warning(f"Invalid score {score}, clamping to 1-5")
        score = max(1, min(5, score))

    return score, reasoning


# =============================================================================
# Oracle Evaluation
# =============================================================================


def run_oracle_eval(
    config: OracleConfig,
    questions: list[str],
    contexts: list[str],
) -> OracleEvalResults:
    """Run activation oracle evaluation on contexts with LLM judge scoring.

    For each (question, context) pair:
    1. Collect activations from the last token of the context
    2. Ask the oracle the question about those activations
    3. Have an LLM judge score the response accuracy (1-5)

    Args:
        config: Oracle configuration
        questions: Questions to ask the oracle
        contexts: Text contexts to collect activations from

    Returns:
        OracleEvalResults with individual results and mean score
    """
    # Validate inputs
    assert config is not None, "Config cannot be None"
    assert (
        isinstance(questions, list) and len(questions) > 0
    ), "Questions must be a non-empty list"
    assert (
        isinstance(contexts, list) and len(contexts) > 0
    ), "Contexts must be a non-empty list"
    assert all(
        isinstance(q, str) and len(q) > 0 for q in questions
    ), "All questions must be non-empty strings"
    assert all(
        isinstance(c, str) and len(c) > 0 for c in contexts
    ), "All contexts must be non-empty strings"

    assert len(config.model_name) > 0, "model_name cannot be empty"
    assert len(config.oracle_path) > 0, "oracle_path cannot be empty"
    assert (
        0 < config.layer_percent <= 100
    ), f"layer_percent must be between 0 and 100, got {config.layer_percent}"
    assert (
        config.max_new_tokens > 0
    ), f"max_new_tokens must be positive, got {config.max_new_tokens}"
    assert (
        config.batch_size > 0
    ), f"batch_size must be positive, got {config.batch_size}"
    assert (
        config.temperature >= 0.0
    ), f"temperature must be non-negative, got {config.temperature}"
    assert len(config.device) > 0, "device cannot be empty"
    assert len(config.dtype) > 0, "dtype cannot be empty"
    assert len(config.judge_model) > 0, "judge_model cannot be empty"
    assert (
        config.judge_max_tokens > 0
    ), f"judge_max_tokens must be positive, got {config.judge_max_tokens}"

    logger.info(f"Loading model: {config.model_name}")

    # Set up device and dtype
    dtype = get_dtype(config.dtype)
    device = torch.device(config.device)
    torch.set_grad_enabled(False)

    # Load model and tokenizer
    tokenizer = load_tokenizer(config.model_name)
    assert tokenizer is not None, "Failed to load tokenizer"
    base_model = load_model(config.model_name, dtype)
    assert base_model is not None, "Failed to load model"

    # Convert to PeftModel for adapter support (accepts AutoModelForCausalLM)
    adapter_name = "oracle"
    # AutoModelForCausalLM is a subclass of torch.nn.Module, but type checker needs help
    model = PeftModel.from_pretrained(
        cast("torch.nn.Module", base_model),
        config.oracle_path,
        adapter_name=adapter_name,
    )
    model.eval()
    # Bug in PEFT where loading an adapter with from_pretrained doesn't set _hf_peft_config_loaded
    base_model._hf_peft_config_loaded = True  # type: ignore[attr-defined]

    # Load oracle adapter (already loaded above as "oracle")
    logger.info(f"Loaded oracle: {config.oracle_path}")

    # Load target adapter if specified (works with PeftModel)
    target_name = None
    if config.target_adapter_path is not None:
        assert (
            len(config.target_adapter_path) > 0
        ), "target_adapter_path cannot be empty"
        logger.info(f"Loading target adapter: {config.target_adapter_path}")
        target_name = load_lora_adapter(base_model, config.target_adapter_path)
        assert (
            target_name is not None and target_name in model.peft_config
        ), "Failed to load target adapter"

    # Build VerbalizerEvalConfig
    verbalizer_config = VerbalizerEvalConfig(
        model_name=config.model_name,
        selected_layer_percent=config.layer_percent,
        verbalizer_generation_kwargs={
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else 1.0,
            "max_new_tokens": config.max_new_tokens,
        },
        # Only use last token activation
        token_start_idx=-1,
        token_end_idx=0,
        verbalizer_input_types=["tokens"],
        activation_input_types=["orig"] if target_name is None else ["lora"],
        eval_batch_size=config.batch_size,
    )

    # Build verbalizer inputs for all (question, context) pairs
    verbalizer_inputs: list[VerbalizerInputInfo] = []
    input_metadata: list[tuple[str, str]] = []  # (context, question) pairs

    for question in questions:
        assert len(question) > 0, "Question cannot be empty"
        for context in contexts:
            assert len(context) > 0, "Context cannot be empty"
            # Format context as chat message
            context_messages: list[dict[str, str]] = [
                {"role": "user", "content": context}
            ]

            verbalizer_inputs.append(
                VerbalizerInputInfo(
                    context_prompt=context_messages,
                    verbalizer_prompt=question,
                    ground_truth="",  # We use LLM judge, not exact match
                )
            )
            input_metadata.append((context, question))

    total_pairs = len(verbalizer_inputs)
    expected_pairs = len(questions) * len(contexts)
    assert (
        total_pairs == expected_pairs
    ), f"Expected {expected_pairs} pairs, got {total_pairs}"
    assert total_pairs > 0, "Must have at least one pair"
    logger.info(f"Running oracle on {total_pairs} (question, context) pairs...")

    # Fix for Gemma3 models: nl_probes expects model.language_model but Gemma3 uses model.model
    # The error trace shows: PeftModel -> base_model -> model.language_model
    # So we need to patch the base_model's underlying model (Gemma3ForCausalLM)
    if not hasattr(base_model, "language_model") and hasattr(base_model, "model"):
        # Add language_model as an alias to model for nl_probes compatibility
        base_model.language_model = base_model.model  # type: ignore[attr-defined]

    verbalizer_results = run_verbalizer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_inputs,
        verbalizer_lora_path=adapter_name,
        target_lora_path=target_name,
        config=verbalizer_config,
        device=device,
    )

    # Process results and run LLM judge
    assert (
        len(verbalizer_results) == total_pairs
    ), f"Expected {total_pairs} verbalizer results, got {len(verbalizer_results)}"
    all_results: list[OracleResult] = []

    for i, vr in enumerate(verbalizer_results):
        assert i < len(input_metadata), f"Index {i} out of range for input_metadata"
        context, question = input_metadata[i]
        assert len(context) > 0, "Context cannot be empty"
        assert len(question) > 0, "Question cannot be empty"

        # Get response - prefer token responses, then segment, then full_seq
        response = ""
        if vr.token_responses:
            reversed_token_responses = (
                response
                for response in reversed(vr.token_responses)
                if response is not None
            )
            response = next(reversed_token_responses, "")
        elif vr.segment_responses:
            response = vr.segment_responses[-1] if vr.segment_responses else ""
        elif vr.full_sequence_responses:
            response = (
                vr.full_sequence_responses[-1] if vr.full_sequence_responses else ""
            )

        if not response:
            logger.warning(f"Empty response for pair {i}")
            response = "(no response)"

        assert len(response) > 0, "Response cannot be empty"

        # Judge the response using the LLM judge
        judge_score, judge_reasoning = judge_response(
            context=context,
            question=question,
            response=response,
            model=config.judge_model,
            max_tokens=config.judge_max_tokens,
            temperature=config.judge_temperature,
        )

        assert 1 <= judge_score <= 5, f"Judge score must be 1-5, got {judge_score}"
        assert len(judge_reasoning) > 0, "Judge reasoning cannot be empty"

        result = OracleResult(
            context=context,
            question=question,
            oracle_response=response,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
        )
        all_results.append(result)

        # Log progress
        if len(all_results) % 10 == 0:
            logger.info(f"Processed {len(all_results)}/{total_pairs} pairs")

    # Cleanup adapters
    if target_name and target_name in model.peft_config:
        model.delete_adapter(target_name)
    if adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)

    # Create aggregated results
    assert len(all_results) > 0, "Must have at least one result"
    assert len(all_results) == len(questions) * len(
        contexts
    ), f"Expected {len(questions) * len(contexts)} results, got {len(all_results)}"

    eval_results = OracleEvalResults(config=config, results=all_results)

    mean = eval_results.mean_score()
    assert 1.0 <= mean <= 5.0, f"Mean score must be between 1 and 5, got {mean}"
    logger.info(f"Mean judge score: {mean:.2f}")

    return eval_results


def save_oracle_results(results: OracleEvalResults, output_path: str | Path) -> None:
    """Save oracle evaluation results to JSON file."""
    assert (
        results is not None
        and hasattr(results, "results")
        and hasattr(results, "config")
    ), "Results must have required attributes"
    assert len(results.results) > 0, "Results cannot be empty"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_score = results.mean_score()
    assert 1.0 <= mean_score <= 5.0, f"Mean score must be 1-5, got {mean_score}"

    cfg = results.config
    data = {
        "mean_score": mean_score,
        "config": {
            "model_name": cfg.model_name if cfg else None,
            "oracle_path": cfg.oracle_path if cfg else None,
            "target_adapter_path": cfg.target_adapter_path if cfg else None,
            "layer_percent": cfg.layer_percent if cfg else None,
            "judge_model": cfg.judge_model if cfg else None,
        },
        "results": [
            {
                "context": r.context,
                "question": r.question,
                "oracle_response": r.oracle_response,
                "judge_score": r.judge_score,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in results.results
        ],
    }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(results.results)} oracle results to {output_path}")


if __name__ == "__main__":
    # Example usage
    from core.questions import get_train_questions

    config = OracleConfig(
        model_name="google/gemma-3-1b-it",
        oracle_path="adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it",
        target_adapter_path=None,
    )

    questions = get_train_questions()[:3]
    contexts = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Machine learning models can learn patterns from data.",
    ]

    results = run_oracle_eval(config, questions, contexts)

    for r in results.results:
        print(f"\nContext: {r.context[:50]}...")
        print(f"Question: {r.question}")
        print(f"Response: {r.oracle_response}")
        print(f"Score: {r.judge_score} - {r.judge_reasoning}")

    print(f"\nMean score: {results.mean_score}")
