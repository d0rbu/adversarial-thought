"""Activation Oracle integration for probing model activations.

This module provides utilities for using activation oracles to query
what information is encoded in model activations at the last token position.
An LLM judge scores the oracle's response accuracy from 1-5.

Setup:
    Set OPENAI_API_KEY environment variable for the LLM judge.

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

import torch
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
    max_new_tokens: int = 100
    temperature: float = 0.0
    do_sample: bool = False

    # Batch size for evaluation
    batch_size: int = 32

    # Device and dtype
    device: str = "cuda"
    dtype: str = "bfloat16"

    # LLM judge settings
    judge_model: str = "gpt-5-nano"
    judge_max_tokens: int = 16384


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

    results: list[OracleResult] = field(default_factory=list)
    config: OracleConfig

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
) -> tuple[int, str]:
    """Use LLM judge to score oracle response accuracy.

    Args:
        context: The original text/prompt
        question: Question asked about the context
        response: Oracle's response to evaluate
        model: OpenAI model to use for judging
        max_tokens: Maximum tokens for judge response

    Returns:
        Tuple of (score 1-5, reasoning) or (None, None) if judging fails
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

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
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Make sure the response did not get truncated
    if completion.choices[0].message.content is None:
        raise ValueError("Response was truncated")

    response_text = completion.choices[0].message.content or ""

    # Parse JSON response
    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    result = json.loads(response_text.strip())
    score = int(result.get("score", 0))
    reasoning = result.get("reasoning", "")

    # Validate score range
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
    logger.info(f"Loading model: {config.model_name}")

    # Set up device and dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)
    device = torch.device(config.device)
    torch.set_grad_enabled(False)

    # Load model and tokenizer
    tokenizer = load_tokenizer(config.model_name)
    model = load_model(config.model_name, dtype)

    # Convert to PeftModel for adapter support (accepts AutoModelForCausalLM)
    model = PeftModel.from_pretrained(model, config.oracle_path, adapter_name="oracle")  # type: ignore[arg-type]
    model.eval()

    # Load oracle adapter (already loaded above as "oracle")
    logger.info(f"Loaded oracle: {config.oracle_path}")
    oracle_name = "oracle"

    # Load target adapter if specified (works with PeftModel)
    target_name = None
    if config.target_adapter_path is not None:
        logger.info(f"Loading target adapter: {config.target_adapter_path}")
        target_name = load_lora_adapter(model, config.target_adapter_path)  # type: ignore[arg-type]

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
        for context in contexts:
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
    logger.info(f"Running oracle on {total_pairs} (question, context) pairs...")

    # Run verbalizer evaluation (works with PeftModel)
    verbalizer_results = run_verbalizer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_inputs,
        verbalizer_lora_path=oracle_name,
        target_lora_path=target_name,
        config=verbalizer_config,
        device=device,
    )

    # Process results and run LLM judge
    all_results: list[OracleResult] = []

    for i, vr in enumerate(verbalizer_results):
        context, question = input_metadata[i]

        # Get response - prefer token responses, then segment, then full_seq
        response = ""
        if vr.token_responses:
            for r in reversed(vr.token_responses):
                if r is not None:
                    response = r
                    break
        elif vr.segment_responses:
            response = vr.segment_responses[-1] if vr.segment_responses else ""
        elif vr.full_sequence_responses:
            response = (
                vr.full_sequence_responses[-1] if vr.full_sequence_responses else ""
            )

        if not response:
            logger.warning(f"Empty response for pair {i}")
            response = "(no response)"

        # Judge the response using the LLM judge
        judge_score, judge_reasoning = judge_response(
            context=context,
            question=question,
            response=response,
            model=config.judge_model,
            max_tokens=config.judge_max_tokens,
        )

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
    if oracle_name in model.peft_config:
        model.delete_adapter(oracle_name)

    # Create aggregated results
    eval_results = OracleEvalResults(results=all_results, config=config)

    logger.info(f"Mean judge score: {eval_results.mean_score():.2f}")

    return eval_results


def save_oracle_results(results: OracleEvalResults, output_path: str | Path) -> None:
    """Save oracle evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "mean_score": results.mean_score,
        "config": {
            "model_name": results.config.model_name if results.config else None,
            "oracle_path": results.config.oracle_path if results.config else None,
            "target_adapter_path": (
                results.config.target_adapter_path if results.config else None
            ),
            "layer_percent": results.config.layer_percent if results.config else None,
            "judge_model": results.config.judge_model if results.config else None,
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
