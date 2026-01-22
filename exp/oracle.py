"""Activation Oracle integration for probing model activations.

This module provides utilities for using activation oracles to query
what information is encoded in model activations. It interfaces with
the activation_oracles submodule.

Usage:
    # From the activation_oracles environment
    from exp.oracle import OracleConfig, run_oracle_eval

    config = OracleConfig(
        model_name="google/gemma-3-1b-it",
        oracle_path="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        target_adapter_path="out/sft_baseline",  # Our finetuned model
    )
    results = run_oracle_eval(config, questions, prompts)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

# Add activation_oracles to path if not already there
ORACLE_DIR = Path(__file__).parent.parent / "activation_oracles"
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))


@dataclass
class OracleConfig:
    """Configuration for activation oracle evaluation."""

    # Base model (must match the oracle's base model)
    model_name: str = "google/gemma-2-9b-it"

    # Oracle (verbalizer) LoRA path on HuggingFace
    # Available oracles: https://huggingface.co/collections/adamkarvonen/activation-oracles
    oracle_path: str = (
        "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"
    )

    # Target adapter path (our finetuned model) - None for base model
    target_adapter_path: str | None = None

    # Activation collection settings
    layer_percent: int = 50  # Which layer to collect activations from (as % of total)
    segment_start: int = -10  # Start of segment to analyze (negative = from end)

    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.0
    do_sample: bool = False

    # Batch size for evaluation
    batch_size: int = 64

    # Device and dtype
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class OracleResult:
    """Result from an oracle query."""

    context_prompt: str
    oracle_question: str
    oracle_response: str
    ground_truth: str | None
    activation_type: str  # "orig", "lora", or "diff"


def load_oracle_dependencies():
    """Import activation oracle dependencies (must be run in oracle env)."""
    try:
        import nl_probes.base_experiment as base_experiment
        from nl_probes.base_experiment import VerbalizerEvalConfig, VerbalizerInputInfo
        from nl_probes.utils.common import load_model, load_tokenizer
        from peft import LoraConfig

        return {
            "base_experiment": base_experiment,
            "VerbalizerEvalConfig": VerbalizerEvalConfig,
            "VerbalizerInputInfo": VerbalizerInputInfo,
            "load_model": load_model,
            "load_tokenizer": load_tokenizer,
            "LoraConfig": LoraConfig,
        }
    except ImportError as e:
        logger.error(
            f"Failed to import activation oracle dependencies: {e}\n"
            "Make sure you're running in the activation_oracles environment:\n"
            "  source activation_oracles/.venv/bin/activate"
        )
        raise


def run_oracle_eval(
    config: OracleConfig,
    oracle_questions: list[str],
    context_prompts: list[str],
    ground_truths: list[str] | None = None,
) -> list[OracleResult]:
    """Run activation oracle evaluation on context prompts.

    Args:
        config: Oracle configuration
        oracle_questions: Questions to ask the oracle about activations
        context_prompts: Prompts to collect activations from
        ground_truths: Optional ground truth answers for evaluation

    Returns:
        List of OracleResult objects with oracle responses
    """
    deps = load_oracle_dependencies()
    base_experiment = deps["base_experiment"]
    VerbalizerEvalConfig = deps["VerbalizerEvalConfig"]
    VerbalizerInputInfo = deps["VerbalizerInputInfo"]
    load_model = deps["load_model"]
    load_tokenizer = deps["load_tokenizer"]
    LoraConfig = deps["LoraConfig"]

    logger.info(f"Loading model: {config.model_name}")

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    device = torch.device(config.device)
    torch.set_grad_enabled(False)

    # Load tokenizer and model
    tokenizer = load_tokenizer(config.model_name)
    model = load_model(config.model_name, dtype)
    model.eval()

    # Add dummy adapter for consistent PeftModel API
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load oracle adapter
    logger.info(f"Loading oracle: {config.oracle_path}")
    oracle_name = base_experiment.load_lora_adapter(model, config.oracle_path)

    # Optionally load target adapter
    target_name = None
    if config.target_adapter_path is not None:
        logger.info(f"Loading target adapter: {config.target_adapter_path}")
        target_name = base_experiment.load_lora_adapter(
            model, config.target_adapter_path
        )

    # Configure evaluation
    generation_kwargs = {
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
    }

    eval_config = VerbalizerEvalConfig(
        model_name=config.model_name,
        activation_input_types=["lora"] if config.target_adapter_path else ["orig"],
        eval_batch_size=config.batch_size,
        verbalizer_generation_kwargs=generation_kwargs,
        full_seq_repeats=1,
        segment_repeats=1,
        segment_start_idx=config.segment_start,
        selected_layer_percent=config.layer_percent,
    )

    # Build prompts
    if ground_truths is None:
        ground_truths = [""] * len(context_prompts)

    verbalizer_prompt_infos: list = []
    for question in oracle_questions:
        for context, truth in zip(context_prompts, ground_truths):
            formatted_prompt = [{"role": "user", "content": context}]
            info = VerbalizerInputInfo(
                context_prompt=formatted_prompt,
                ground_truth=truth,
                verbalizer_prompt=question,
            )
            verbalizer_prompt_infos.append(info)

    # Run evaluation
    logger.info(
        f"Running oracle on {len(verbalizer_prompt_infos)} prompt-question pairs..."
    )
    raw_results = base_experiment.run_verbalizer(
        model=model,
        tokenizer=tokenizer,
        verbalizer_prompt_infos=verbalizer_prompt_infos,
        verbalizer_lora_path=config.oracle_path,
        target_lora_path=config.target_adapter_path,
        config=eval_config,
        device=device,
    )

    # Convert to OracleResult format
    results: list[OracleResult] = []
    for r in raw_results:
        # Get the best response (segment or full_seq)
        response = ""
        if r.segment_responses:
            response = r.segment_responses[0]
        elif r.full_sequence_responses:
            response = r.full_sequence_responses[0]

        # Extract context as string
        context_str = r.context_prompt[0]["content"] if r.context_prompt else ""

        results.append(
            OracleResult(
                context_prompt=context_str,
                oracle_question=r.verbalizer_prompt,
                oracle_response=response,
                ground_truth=r.ground_truth,
                activation_type=r.act_key,
            )
        )

    # Cleanup adapters
    if target_name and target_name in model.peft_config:
        model.delete_adapter(target_name)
    if oracle_name in model.peft_config:
        model.delete_adapter(oracle_name)

    return results


def save_oracle_results(results: list[OracleResult], output_path: str | Path) -> None:
    """Save oracle results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "context_prompt": r.context_prompt,
            "oracle_question": r.oracle_question,
            "oracle_response": r.oracle_response,
            "ground_truth": r.ground_truth,
            "activation_type": r.activation_type,
        }
        for r in results
    ]

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(results)} oracle results to {output_path}")


if __name__ == "__main__":
    # Example usage - run from activation_oracles environment
    from core.questions import get_all_questions

    config = OracleConfig(
        model_name="google/gemma-2-9b-it",
        oracle_path="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        target_adapter_path=None,  # Base model
    )

    # Sample questions from our question set
    questions = get_all_questions()
    sample_questions = [q.text for q in questions.sample_train(n=3, seed=42)]

    # Sample context prompts
    context_prompts = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Machine learning models can learn patterns from data.",
    ]

    results = run_oracle_eval(config, sample_questions, context_prompts)

    for r in results:
        print(f"\nContext: {r.context_prompt[:50]}...")
        print(f"Question: {r.oracle_question}")
        print(f"Response: {r.oracle_response}")
