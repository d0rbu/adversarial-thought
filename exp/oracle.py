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
        model_name="Qwen/Qwen3-8B",
        oracle_path="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        target_adapter_path="out/sft_baseline",
    )
    # Create (context, question) pairs - one question per context
    context_question_pairs = [
        ("The capital of France is Paris.", "What is the topic?"),
        ("Machine learning models can learn patterns.", "What is being described?"),
    ]
    results = run_oracle_eval(config, context_question_pairs)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from dotenv import load_dotenv
from loguru import logger
from nl_probes.base_experiment import (
    VerbalizerEvalConfig,
    VerbalizerInputInfo,
    load_lora_adapter,
    load_model,
    load_tokenizer,
)
from openai import OpenAI
from peft import PeftModel
from transformers import BitsAndBytesConfig

from core.dtype import get_dtype
from core.verbalizer import run_verbalizer

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()


@dataclass
class OracleConfig:
    """Configuration for activation oracle evaluation."""

    # Base model (must match the oracle's base model)
    model_name: str = "Qwen/Qwen3-8B"

    # Oracle LoRA path on HuggingFace
    # Available oracles: https://huggingface.co/collections/adamkarvonen/activation-oracles
    oracle_path: str = (
        "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
    )

    # Target adapter path (our finetuned model) - None for base model
    target_adapter_path: str | None = None

    # Which layer to extract activations from (as % of model depth)
    layer_percent: int = 50

    # Generation settings for oracle
    max_new_tokens: int = 4096
    temperature: float = 0.0
    do_sample: bool = False

    # Batch size for evaluation
    batch_size: int = 32

    # Device and dtype
    device: str = "cuda"
    dtype: str = "float16"
    load_in_8bit: bool = False

    # LLM judge settings
    judge_model: str = "gpt-5-nano"
    judge_max_tokens: int = 4096
    judge_temperature: float = 1.0

    # Token indices for activation extraction
    token_start_idx: int = -10
    token_end_idx: int = 0

    # Repeat count for segment or full_seq input types (applies to whichever is used)
    repeats: int = 10


@dataclass(frozen=True)
class OracleResult:
    """Result from an oracle query with LLM judge score."""

    context: str
    question: str
    oracle_response: str
    judge_score: int  # 1-5 accuracy score from LLM judge
    judge_reasoning: str
    judge_prompt: str  # User message sent to the LLM judge
    verbalizer_prompt: str  # Raw prompt sent to the verbalizer (context + question)

    def __post_init__(self) -> None:
        assert (
            self.judge_score > 0 and self.judge_score <= 5
        ), "Judge score must be between 1 and 5"
        assert len(self.judge_reasoning) > 0, "Judge reasoning cannot be empty"
        assert len(self.judge_prompt) > 0, "Judge prompt cannot be empty"
        assert len(self.verbalizer_prompt) > 0, "Verbalizer prompt cannot be empty"
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


@dataclass(frozen=True)
class OracleResultNoJudge:
    """Result from an oracle query without LLM judge score."""

    context: str
    question: str
    oracle_response: str
    verbalizer_prompt: str  # Raw prompt sent to the verbalizer (context + question)

    def __post_init__(self) -> None:
        assert len(self.verbalizer_prompt) > 0, "Verbalizer prompt cannot be empty"
        assert len(self.oracle_response) > 0, "Oracle response cannot be empty"
        assert len(self.context) > 0, "Context cannot be empty"
        assert len(self.question) > 0, "Question cannot be empty"


@dataclass
class OracleEvalResultsNoJudge:
    """Aggregated results from oracle evaluation without judge scoring."""

    config: OracleConfig
    results: list[OracleResultNoJudge] = field(default_factory=list)


# =============================================================================
# LLM Judge
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing whether an AI's response about text activations is accurate.

You will be given:
1. CONTEXT: The original text that was processed
2. TOKENIZED CONTEXT: How the model tokenizer splits the context into tokens (shown with | separators)
3. QUESTION: A question about the context
4. RESPONSE: An AI's answer to that question based on the text's "activations"

Your task: Score the RESPONSE's accuracy from 1-5 based on how well it answers the QUESTION given the CONTEXT and TOKENIZED CONTEXT.

Scoring rubric:
5 - Completely accurate: Response directly and correctly answers the question
4 - Mostly accurate: Response is largely correct with minor inaccuracies or missing details
3 - Partially accurate: Response contains some correct information but also errors or is vague
2 - Mostly inaccurate: Response is largely wrong or irrelevant
1 - Completely inaccurate: Response is entirely wrong or nonsensical

The AI answering the question is an activation oracle, which is a model that has been fine-tuned to answer natural-language questions about embeddings that come from the internal activations of a target large language model. For these questions, we are giving the last 10 tokens of the context to the oracle, which includes end-of-turn and newline tokens.

Respond in JSON format:
{"reasoning": "<brief explanation>", "score": <1-5>}"""


JUDGE_USER_PROMPT = """
CONTEXT: {context}
TOKENIZED CONTEXT: {tokenized_context}
QUESTION: {question}
RESPONSE: {response}
"""


def format_tokenized_context(context: str, tokenizer: AutoTokenizer | Any) -> str:
    """Format context with token boundaries shown using pipe separators.

    Args:
        context: The original text context
        tokenizer: Tokenizer to use for tokenization

    Returns:
        Formatted string with tokens separated by |, e.g., "The |cap|ital |of |France |is |Paris|."
    """
    # Tokenize the context
    tokens = tokenizer.tokenize(context)
    # Join tokens with pipe separators
    return "|".join(tokens)


NUM_ATTEMPTS = 3


def judge_response(
    context: str,
    question: str,
    response: str,
    tokenizer: AutoTokenizer | Any,
    model: str = "gpt-5-nano",
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> tuple[int, str, str]:
    """Use LLM judge to score oracle response accuracy.

    Args:
        context: The original text/prompt
        question: Question asked about the context
        response: Oracle's response to evaluate
        tokenizer: Tokenizer to use for creating tokenized context
        model: OpenAI model to use for judging
        max_tokens: Maximum tokens for judge response
        temperature: Temperature for judge model

    Returns:
        Tuple of (score 1-5, reasoning, user_prompt)
        user_prompt is the user message string sent to the judge
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
    assert tokenizer is not None, "Tokenizer cannot be None"

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY environment variable not set"

    # Format tokenized context
    tokenized_context = format_tokenized_context(context, tokenizer)

    client = OpenAI(api_key=api_key)
    user_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        tokenized_context=tokenized_context,
        question=question,
        response=response,
    )

    for attempt_idx in range(NUM_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            response_text = completion.choices[0].message.content

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

            break
        except Exception as e:
            logger.warning(f"Attempt {attempt_idx + 1} failed: {e}")
            continue
    else:
        should_continue = input(
            f"Failed to get response after {NUM_ATTEMPTS} attempts, prompt:\n{user_prompt}\n\nContinue? (Y/n)"
        )
        if should_continue.strip()[0].lower() == "n":
            raise ValueError("Failed to get response after multiple attempts")
        else:
            score = int(input("Enter score (1-5): "))
            reasoning = "Manual intervention required"

    return score, reasoning, user_prompt


# =============================================================================
# Oracle Evaluation
# =============================================================================

ORACLE_ADAPTER_NAME = "oracle"


def _load_oracle_model(
    config: OracleConfig,
) -> tuple[PeftModel, AutoTokenizer, Any, torch.device]:
    """Load oracle model and tokenizer.

    Args:
        config: Oracle configuration

    Returns:
        Tuple of (model, tokenizer, dtype, device)
    """
    logger.info(f"Loading model: {config.model_name}")

    # Set up device and dtype
    dtype = get_dtype(config.dtype)
    device = torch.device(config.device)
    torch.set_grad_enabled(False)

    # Load model and tokenizer
    tokenizer = load_tokenizer(config.model_name)
    assert tokenizer is not None, "Failed to load tokenizer"

    # Load model with optional 8-bit quantization
    model_kwargs: dict[str, Any] = {}
    if config.load_in_8bit:
        logger.info("Loading model in 8-bit mode")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto" if device.type == "cuda" else None
        model_kwargs["trust_remote_code"] = True

    base_model: AutoModelForCausalLM = load_model(
        config.model_name, dtype, **model_kwargs
    )
    assert base_model is not None, "Failed to load model"

    model = PeftModel.from_pretrained(
        base_model,  # type: ignore[arg-type]
        config.oracle_path,
        adapter_name=ORACLE_ADAPTER_NAME,
    )
    model.eval()

    model.set_adapter(ORACLE_ADAPTER_NAME)
    base_model._hf_peft_config_loaded = True  # type: ignore[attr-defined]

    # Fix for Gemma3 models: nl_probes expects model.language_model but Gemma3 uses model.model
    if not hasattr(base_model, "language_model") and hasattr(base_model, "model"):
        base_model.language_model = base_model.model  # type: ignore[attr-defined]

    # Fix for Qwen3 models: nl_probes expects model.model.layers but Qwen3 has model.model.model.layers
    is_qwen3 = "qwen" in config.model_name.lower() and "3" in config.model_name.lower()
    if is_qwen3 and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        peft_base = model.base_model
        peft_model_model = peft_base.model
        if not hasattr(peft_model_model, "layers") and hasattr(
            peft_model_model, "model"
        ):
            inner_model = peft_model_model.model
            if hasattr(inner_model, "layers"):
                peft_model_model.layers = inner_model.layers  # type: ignore[attr-defined]

    return model, tokenizer, dtype, device


def run_oracle_eval(
    config: OracleConfig,
    context_question_pairs: list[tuple[str, str]],
) -> OracleEvalResults:
    """Run activation oracle evaluation on contexts with LLM judge scoring.

    For each (context, question) pair:
    1. Collect activations from the last token of the context
    2. Ask the oracle the question about those activations
    3. Have an LLM judge score the response accuracy (1-5)

    Args:
        config: Oracle configuration
        context_question_pairs: List of (context, question) tuples to evaluate

    Returns:
        OracleEvalResults with individual results and mean score
    """
    # Validate inputs
    assert config is not None, "Config cannot be None"
    assert (
        isinstance(context_question_pairs, list) and len(context_question_pairs) > 0
    ), "context_question_pairs must be a non-empty list"
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in context_question_pairs
    ), "All pairs must be tuples of length 2"
    assert all(
        isinstance(c, str) and len(c) > 0 and isinstance(q, str) and len(q) > 0
        for c, q in context_question_pairs
    ), "All contexts and questions must be non-empty strings"

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

    # Load model with optional 8-bit quantization
    model_kwargs: dict[str, Any] = {}
    if config.load_in_8bit:
        logger.info("Loading model in 8-bit mode")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto" if device.type == "cuda" else None
        model_kwargs["trust_remote_code"] = True

    base_model: AutoModelForCausalLM = load_model(
        config.model_name, dtype, **model_kwargs
    )
    assert base_model is not None, "Failed to load model"

    model = PeftModel.from_pretrained(
        base_model,  # type: ignore[arg-type]
        config.oracle_path,
        adapter_name=ORACLE_ADAPTER_NAME,
    )
    model.eval()

    model.set_adapter(ORACLE_ADAPTER_NAME)
    base_model._hf_peft_config_loaded = True  # type: ignore[attr-defined]

    # Load target adapter if specified
    target_name = None
    if config.target_adapter_path is not None:
        assert (
            len(config.target_adapter_path) > 0
        ), "target_adapter_path cannot be empty"
        logger.info(f"Loading target adapter: {config.target_adapter_path}")
        # load_lora_adapter expects AutoModelForCausalLM but accepts PeftModel in practice
        # Pass the base model instead, or use type ignore
        target_name = load_lora_adapter(model, config.target_adapter_path)  # type: ignore[arg-type]
        assert (
            target_name is not None and target_name in model.peft_config
        ), "Failed to load target adapter"

    # Determine verbalizer input type based on token indices
    # If token_end_idx - token_start_idx == 1: use tokens
    # If token_end_idx - token_start_idx > 1: use segment
    # If token_end_idx - token_start_idx == 0: use full_seq
    token_diff = config.token_end_idx - config.token_start_idx
    if token_diff == 0:
        verbalizer_input_types = ["full_seq"]
        # For full_seq, we don't need segment indices, but set them to avoid errors
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "full_seq"
        segment_repeats = 0  # Not used for full_seq
        full_seq_repeats = config.repeats
    elif token_diff == 1 or token_diff == -1:
        verbalizer_input_types = ["tokens"]
        # For tokens, we don't need segment indices, but set them to avoid errors
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "tokens"
        segment_repeats = 0  # Not used for tokens
        full_seq_repeats = 0  # Not used for tokens
    else:  # token_diff > 1
        verbalizer_input_types = ["segment"]
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "segment"
        segment_repeats = config.repeats
        full_seq_repeats = 0  # Not used for segment

    logger.info(
        f"Using verbalizer input type: {preferred_response_type} "
        f"(token_start_idx={config.token_start_idx}, token_end_idx={config.token_end_idx}, diff={token_diff}, repeats={config.repeats})"
    )

    # Build VerbalizerEvalConfig
    verbalizer_config = VerbalizerEvalConfig(
        model_name=config.model_name,
        selected_layer_percent=config.layer_percent,
        verbalizer_generation_kwargs={
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else 1.0,
            "max_new_tokens": config.max_new_tokens,
        },
        token_start_idx=config.token_start_idx,
        token_end_idx=config.token_end_idx,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        segment_repeats=segment_repeats,
        full_seq_repeats=full_seq_repeats,
        add_generation_prompt=False,
        enable_thinking=False,
        verbalizer_input_types=verbalizer_input_types,
        activation_input_types=["orig"] if target_name is None else ["lora"],
        eval_batch_size=config.batch_size,
    )

    # Build verbalizer inputs for all (context, question) pairs
    verbalizer_inputs: list[VerbalizerInputInfo] = []
    input_metadata: list[tuple[str, str]] = []  # (context, question) pairs

    for context, question in context_question_pairs:
        assert len(context) > 0, "Context cannot be empty"
        assert len(question) > 0, "Question cannot be empty"
        # Format context as chat message
        context_messages: list[dict[str, str]] = [{"role": "user", "content": context}]

        verbalizer_inputs.append(
            VerbalizerInputInfo(
                context_prompt=context_messages,
                verbalizer_prompt=question,
                ground_truth="",  # We use LLM judge, not exact match
            )
        )
        input_metadata.append((context, question))

    total_pairs = len(verbalizer_inputs)
    assert total_pairs == len(
        context_question_pairs
    ), f"Expected {len(context_question_pairs)} pairs, got {total_pairs}"
    assert total_pairs > 0, "Must have at least one pair"
    logger.info(f"Running oracle on {total_pairs} (context, question) pairs...")

    # Fix for Gemma3 models: nl_probes expects model.language_model but Gemma3 uses model.model
    # The error trace shows: PeftModel -> base_model -> model.language_model
    # So we need to patch the base_model's underlying model (Gemma3ForCausalLM)
    if not hasattr(base_model, "language_model") and hasattr(base_model, "model"):
        # Add language_model as an alias to model for nl_probes compatibility
        base_model.language_model = base_model.model  # type: ignore[attr-defined]

    # Fix for Qwen3 models: nl_probes expects model.model.layers but Qwen3 has model.model.model.layers
    # We need to patch AFTER PeftModel is created because get_hf_submodule receives the PeftModel
    is_qwen3 = "qwen" in config.model_name.lower() and "3" in config.model_name.lower()
    # Patch on the PeftModel's base_model (which is what get_hf_submodule will access)
    if is_qwen3 and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        peft_base = model.base_model
        peft_model_model = peft_base.model
        # If peft_base.model doesn't have .layers but has .model.layers, patch it
        if not hasattr(peft_model_model, "layers") and hasattr(
            peft_model_model, "model"
        ):
            inner_model = peft_model_model.model
            if hasattr(inner_model, "layers"):
                # Add .layers attribute that points to .model.layers
                peft_model_model.layers = inner_model.layers  # type: ignore[attr-defined]

    verbalizer_results = run_verbalizer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_inputs,
        verbalizer_lora_path=ORACLE_ADAPTER_NAME,
        target_lora_path=target_name,
        config=verbalizer_config,
        device=device,
        dtype=dtype,  # Pass dtype parameter to use our fixed version
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

        # Construct the raw verbalizer prompt
        # The verbalizer receives the question as a string prompt, and the context
        # is used separately to collect activations. We format both here for clarity.
        # Format context as text using chat template (just the context)
        context_messages: list[dict[str, str]] = [{"role": "user", "content": context}]
        context_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            context_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(context_text, str), "Context text must be a string"
        # The verbalizer prompt is just the question, but we show context for reference
        verbalizer_prompt = f"Context: {context_text}\n\nQuestion: {question}"
        assert len(verbalizer_prompt) > 0, "Verbalizer prompt cannot be empty"

        # Get response - check all types (only one will have responses based on configured input type)
        # Priority: segment > tokens > full_seq (segment is typically most informative)
        if vr.segment_responses:
            response = vr.segment_responses[-1]
        elif vr.token_responses:
            reversed_token_responses = (
                response
                for response in reversed(vr.token_responses)
                if response is not None
            )
            response = next(reversed_token_responses)
        elif vr.full_sequence_responses:
            response = vr.full_sequence_responses[-1]
        else:
            raise ValueError(f"No response found for pair {i}")

        if not response:
            logger.warning(f"Empty response for pair {i}")
            response = "(no response)"

        assert len(response) > 0, "Response cannot be empty"

        # Judge the response using the LLM judge
        judge_score, judge_reasoning, judge_prompt = judge_response(
            context=context,
            question=question,
            response=response,
            tokenizer=tokenizer,
            model=config.judge_model,
            max_tokens=config.judge_max_tokens,
            temperature=config.judge_temperature,
        )

        assert 1 <= judge_score <= 5, f"Judge score must be 1-5, got {judge_score}"
        assert len(judge_reasoning) > 0, "Judge reasoning cannot be empty"
        assert len(judge_prompt) > 0, "Judge prompt cannot be empty"

        result = OracleResult(
            context=context,
            question=question,
            oracle_response=response,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            judge_prompt=judge_prompt,
            verbalizer_prompt=verbalizer_prompt,
        )
        all_results.append(result)

        # Log progress
        if len(all_results) % 10 == 0:
            logger.info(f"Processed {len(all_results)}/{total_pairs} pairs")

    # Cleanup adapters
    if target_name and target_name in model.peft_config:
        model.delete_adapter(target_name)
    if ORACLE_ADAPTER_NAME in model.peft_config:
        model.delete_adapter(ORACLE_ADAPTER_NAME)

    # Create aggregated results
    assert len(all_results) > 0, "Must have at least one result"
    assert len(all_results) == len(
        context_question_pairs
    ), f"Expected {len(context_question_pairs)} results, got {len(all_results)}"

    eval_results = OracleEvalResults(config=config, results=all_results)

    mean = eval_results.mean_score()
    assert 1.0 <= mean <= 5.0, f"Mean score must be between 1 and 5, got {mean}"
    logger.info(f"Mean judge score: {mean:.2f}")

    return eval_results


def run_oracle_eval_no_judge(
    config: OracleConfig,
    context_question_pairs: list[tuple[str, str]],
    model: PeftModel | None = None,
    tokenizer: AutoTokenizer | None = None,
    dtype: Any | None = None,
    device: torch.device | None = None,
    cleanup_adapters: bool = True,
) -> OracleEvalResultsNoJudge:
    """Run activation oracle evaluation on contexts WITHOUT LLM judge scoring.

    For each (context, question) pair:
    1. Collect activations from the last token of the context
    2. Ask the oracle the question about those activations
    3. Return the response (no judge scoring)

    Args:
        config: Oracle configuration
        context_question_pairs: List of (context, question) tuples to evaluate
        model: Optional pre-loaded PeftModel. If None, model will be loaded.
        tokenizer: Optional pre-loaded tokenizer. If None, tokenizer will be loaded.
        dtype: Optional dtype. If None, will be derived from config.
        device: Optional device. If None, will be derived from config.
        cleanup_adapters: Whether to cleanup adapters at the end. Set to False when
            reusing the model across multiple calls.

    Returns:
        OracleEvalResultsNoJudge with individual results (no judge scores)
    """
    # Validate inputs
    assert config is not None, "Config cannot be None"
    assert (
        isinstance(context_question_pairs, list) and len(context_question_pairs) > 0
    ), "context_question_pairs must be a non-empty list"
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in context_question_pairs
    ), "All pairs must be tuples of length 2"
    assert all(
        isinstance(c, str) and len(c) > 0 and isinstance(q, str) and len(q) > 0
        for c, q in context_question_pairs
    ), "All contexts and questions must be non-empty strings"

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

    # Load model and tokenizer if not provided
    model_loaded_here = False
    if model is None or tokenizer is None:
        model, tokenizer, dtype, device = _load_oracle_model(config)
        model_loaded_here = True
    else:
        # Use provided model/tokenizer, but ensure dtype and device are set
        if dtype is None:
            dtype = get_dtype(config.dtype)
        if device is None:
            device = torch.device(config.device)
        torch.set_grad_enabled(False)

    # Load target adapter if specified
    target_name = None
    if config.target_adapter_path is not None:
        assert (
            len(config.target_adapter_path) > 0
        ), "target_adapter_path cannot be empty"
        logger.info(f"Loading target adapter: {config.target_adapter_path}")
        # load_lora_adapter expects AutoModelForCausalLM but accepts PeftModel in practice
        # Pass the base model instead, or use type ignore
        target_name = load_lora_adapter(model, config.target_adapter_path)  # type: ignore[arg-type]
        assert (
            target_name is not None and target_name in model.peft_config
        ), "Failed to load target adapter"

    # Determine verbalizer input type based on token indices
    # If token_end_idx - token_start_idx == 1: use tokens
    # If token_end_idx - token_start_idx > 1: use segment
    # If token_end_idx - token_start_idx == 0: use full_seq
    token_diff = config.token_end_idx - config.token_start_idx
    if token_diff == 0:
        verbalizer_input_types = ["full_seq"]
        # For full_seq, we don't need segment indices, but set them to avoid errors
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "full_seq"
        segment_repeats = 0  # Not used for full_seq
        full_seq_repeats = config.repeats
    elif token_diff == 1 or token_diff == -1:
        verbalizer_input_types = ["tokens"]
        # For tokens, we don't need segment indices, but set them to avoid errors
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "tokens"
        segment_repeats = 0  # Not used for tokens
        full_seq_repeats = 0  # Not used for tokens
    else:  # token_diff > 1
        verbalizer_input_types = ["segment"]
        segment_start_idx = config.token_start_idx
        segment_end_idx = config.token_end_idx
        preferred_response_type = "segment"
        segment_repeats = config.repeats
        full_seq_repeats = 0  # Not used for segment

    logger.info(
        f"Using verbalizer input type: {preferred_response_type} "
        f"(token_start_idx={config.token_start_idx}, token_end_idx={config.token_end_idx}, diff={token_diff}, repeats={config.repeats})"
    )

    # Build VerbalizerEvalConfig
    verbalizer_config = VerbalizerEvalConfig(
        model_name=config.model_name,
        selected_layer_percent=config.layer_percent,
        verbalizer_generation_kwargs={
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else 1.0,
            "max_new_tokens": config.max_new_tokens,
        },
        token_start_idx=config.token_start_idx,
        token_end_idx=config.token_end_idx,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        segment_repeats=segment_repeats,
        full_seq_repeats=full_seq_repeats,
        add_generation_prompt=False,
        enable_thinking=False,
        verbalizer_input_types=verbalizer_input_types,
        activation_input_types=["orig"] if target_name is None else ["lora"],
        eval_batch_size=config.batch_size,
    )

    # Build verbalizer inputs for all (context, question) pairs
    verbalizer_inputs: list[VerbalizerInputInfo] = []
    input_metadata: list[tuple[str, str]] = []  # (context, question) pairs

    for context, question in context_question_pairs:
        assert len(context) > 0, "Context cannot be empty"
        assert len(question) > 0, "Question cannot be empty"
        # Format context as chat message
        context_messages: list[dict[str, str]] = [{"role": "user", "content": context}]

        verbalizer_inputs.append(
            VerbalizerInputInfo(
                context_prompt=context_messages,
                verbalizer_prompt=question,
                ground_truth="",  # Not used when no judge
            )
        )
        input_metadata.append((context, question))

    total_pairs = len(verbalizer_inputs)
    assert total_pairs == len(
        context_question_pairs
    ), f"Expected {len(context_question_pairs)} pairs, got {total_pairs}"
    assert total_pairs > 0, "Must have at least one pair"
    logger.info(
        f"Running oracle on {total_pairs} (context, question) pairs (no judge)..."
    )

    verbalizer_results = run_verbalizer(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_inputs,
        verbalizer_lora_path=ORACLE_ADAPTER_NAME,
        target_lora_path=target_name,
        config=verbalizer_config,
        device=device,
        dtype=dtype,  # Pass dtype parameter to use our fixed version
    )

    # Process results WITHOUT judge scoring
    assert (
        len(verbalizer_results) == total_pairs
    ), f"Expected {total_pairs} verbalizer results, got {len(verbalizer_results)}"
    all_results: list[OracleResultNoJudge] = []

    for i, vr in enumerate(verbalizer_results):
        assert i < len(input_metadata), f"Index {i} out of range for input_metadata"
        context, question = input_metadata[i]
        assert len(context) > 0, "Context cannot be empty"
        assert len(question) > 0, "Question cannot be empty"

        # Construct the raw verbalizer prompt
        # The verbalizer receives the question as a string prompt, and the context
        # is used separately to collect activations. We format both here for clarity.
        # Format context as text using chat template (just the context)
        context_messages: list[dict[str, str]] = [{"role": "user", "content": context}]
        context_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            context_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(context_text, str), "Context text must be a string"
        # The verbalizer prompt is just the question, but we show context for reference
        verbalizer_prompt = f"Context: {context_text}\n\nQuestion: {question}"
        assert len(verbalizer_prompt) > 0, "Verbalizer prompt cannot be empty"

        # Get response - check all types (only one will have responses based on configured input type)
        # Priority: segment > tokens > full_seq (segment is typically most informative)
        if vr.segment_responses:
            response = vr.segment_responses[-1]
        elif vr.token_responses:
            reversed_token_responses = (
                response
                for response in reversed(vr.token_responses)
                if response is not None
            )
            response = next(reversed_token_responses)
        elif vr.full_sequence_responses:
            response = vr.full_sequence_responses[-1]
        else:
            raise ValueError(f"No response found for pair {i}")

        if not response:
            logger.warning(f"Empty response for pair {i}")
            response = "(no response)"

        assert len(response) > 0, "Response cannot be empty"

        # Create result WITHOUT judge scoring
        result = OracleResultNoJudge(
            context=context,
            question=question,
            oracle_response=response,
            verbalizer_prompt=verbalizer_prompt,
        )
        all_results.append(result)

        # Log progress
        if len(all_results) % 10 == 0:
            logger.info(f"Processed {len(all_results)}/{total_pairs} pairs")

    # Cleanup adapters only if requested and model was loaded here
    if cleanup_adapters:
        if target_name and target_name in model.peft_config:
            model.delete_adapter(target_name)
        # Only delete oracle adapter if we loaded the model here
        if model_loaded_here and ORACLE_ADAPTER_NAME in model.peft_config:
            model.delete_adapter(ORACLE_ADAPTER_NAME)

    # Create aggregated results
    assert len(all_results) > 0, "Must have at least one result"
    assert len(all_results) == len(
        context_question_pairs
    ), f"Expected {len(context_question_pairs)} results, got {len(all_results)}"

    eval_results = OracleEvalResultsNoJudge(config=config, results=all_results)

    logger.info(
        f"Completed oracle evaluation on {len(all_results)} pairs (no judge scoring)"
    )

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
                "judge_prompt": r.judge_prompt,
                "verbalizer_prompt": r.verbalizer_prompt,
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
        model_name="Qwen/Qwen3-8B",
        oracle_path="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        target_adapter_path=None,
    )

    questions = get_train_questions()[:3]
    contexts = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Machine learning models can learn patterns from data.",
    ]

    # Create pairs: assign one question to each context
    import random

    random.seed(42)
    context_question_pairs = [
        (context, random.choice(questions)) for context in contexts
    ]

    results = run_oracle_eval(config, context_question_pairs)

    for r in results.results:
        print(f"\nContext: {r.context[:50]}...")
        print(f"Question: {r.question}")
        print(f"Response: {r.oracle_response}")
        print(f"Score: {r.judge_score} - {r.judge_reasoning}")

    print(f"\nMean score: {results.mean_score}")
