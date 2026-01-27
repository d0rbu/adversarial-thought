"""Adversarial Supervised Fine-Tuning (SFT) experiment.

This module implements adversarial SFT training where we:
1. Generate a dataset of (document, question, oracle_answer) triples
2. During training, compute both SFT loss and adversarial loss
3. Adversarial loss maximizes oracle error on answer tokens
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as t
from datasets import Dataset
from loguru import logger

# Import nl_probes utilities
from nl_probes.utils.activation_utils import (
    get_hf_submodule,
)
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import (
    SPECIAL_TOKEN,
    find_pattern_in_tokens,
    get_introspection_prefix,
)
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

import hydra
import wandb
from core.data import load_and_prepare_conversation_dataset
from core.dtype import get_dtype
from core.questions import get_train_questions
from core.steering_hooks import add_hook, get_hf_activation_steering_hook
from exp.oracle import OracleConfig, _load_oracle_model, run_oracle_eval_no_judge


@dataclass
class AdversarialExperimentConfig:
    """Flattened experiment configuration for adversarial SFT."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    tokenizer_name: str = "Qwen/Qwen3-8B"

    # LoRA
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Data
    dataset_name: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 2048
    train_ratio: float = 0.9
    max_samples: int | None = None
    training_subset_size: int | None = None  # Limit adversarial training dataset size
    max_messages_per_conversation: int = 3

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # Adversarial
    adversarial_alpha: float = 1.0
    oracle_path: str = (
        "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
    )
    layer_percent: int = 50
    token_start_idx: int = -10
    token_end_idx: int = -2
    repeats: int = 10
    question_seed: int = 42

    # Hardware
    dtype: str = "float16"
    device: str = "cuda"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    attn_implementation: str = "sdpa"

    # Experiment
    seed: int = 42
    output_dir: str = "out/adversarial_sft"
    cache_dir: str = ".cache/adv-thought"
    cache_batch_size: int = 32  # Batch size for incremental cache generation

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "adversarial-thought"


ORACLE_ADAPTER_NAME = "oracle"
TRAINABLE_ADAPTER_NAME = "trainable"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    cfg: AdversarialExperimentConfig,
) -> tuple[Any, PreTrainedTokenizer]:
    """Load the model and tokenizer with optional LoRA.

    Returns a tuple of (model, tokenizer). The model may be a PreTrainedModel
    or a PeftModel if LoRA is enabled.
    """
    logger.info(f"Loading model: {cfg.model_name}")

    # Determine dtype
    torch_dtype = get_dtype(cfg.dtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up quantization config if needed
    quantization_config = None
    if cfg.load_in_4bit:
        logger.info("Loading model with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif cfg.load_in_8bit:
        logger.info("Loading model with 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch_dtype,
        )

    # Load model
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if cfg.device == "cuda" else None,
        "trust_remote_code": True,
        "attn_implementation": cfg.attn_implementation,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **model_kwargs,
    )

    # Apply LoRA if enabled
    if cfg.lora_enabled:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def _get_cache_key(document: str, question: str) -> str:
    """Generate a cache key for a (document, question) pair."""
    content = f"{document}|||{question}"
    return hashlib.sha256(content.encode()).hexdigest()


def _load_cache(cache_dir: Path) -> dict[str, dict[str, str]]:
    """Load cached triplets from disk.

    Returns:
        Dictionary mapping cache_key to {document, question, oracle_answer}
    """
    cache_file = cache_dir / "adversarial_dataset_cache.json"
    if not cache_file.exists():
        return {}

    try:
        with cache_file.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load cache: {e}. Starting fresh.")
        return {}


def _save_cache(cache_dir: Path, cache: dict[str, dict[str, str]]) -> None:
    """Save cache to disk periodically.

    Args:
        cache_dir: Directory to save cache
        cache: Cache dictionary to save
    """
    cache_file = cache_dir / "adversarial_dataset_cache.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save to temporary file first, then rename (atomic write)
    temp_file = cache_file.with_suffix(".tmp")
    try:
        with temp_file.open("w") as f:
            json.dump(cache, f, indent=2)
        temp_file.replace(cache_file)
    except OSError as e:
        logger.warning(f"Failed to save cache: {e}")


def generate_adversarial_dataset(
    datasets: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cfg: AdversarialExperimentConfig,
) -> list[dict[str, Any]]:
    """Generate adversarial dataset with (document, question, oracle_answer) triples.

    This function is resumable - it caches triplets to disk incrementally and can
    resume from where it left off if interrupted. It processes pairs in batches
    and saves results after each batch.

    Args:
        datasets: DatasetDict with train/validation splits
        tokenizer: Tokenizer for formatting
        cfg: Experiment configuration

    Returns:
        List of dicts with keys: document, question, oracle_answer, input_ids, attention_mask
    """
    logger.info("Generating adversarial dataset...")

    # Set up cache directory
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get training questions
    train_questions = get_train_questions()
    logger.info(f"Using {len(train_questions)} training questions")

    # Get subset of training data
    train_dataset = datasets["train"]
    if cfg.max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(cfg.max_samples, len(train_dataset)))
        )

    # Set seed for question selection
    rng = random.Random(cfg.question_seed)

    # Prepare all context-question pairs with their cache keys
    all_pairs_with_keys: list[
        tuple[str, str, str]
    ] = []  # (document, question, cache_key)

    for example in train_dataset:
        # Extract document text from tokenized example
        # The dataset is already tokenized, so we need to decode it
        input_ids = example["input_ids"]
        document_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        if len(document_text.strip()) == 0:
            continue

        # Randomly select one question per document
        question = rng.choice(train_questions)
        cache_key = _get_cache_key(document_text, question)
        all_pairs_with_keys.append((document_text, question, cache_key))

    logger.info(f"Prepared {len(all_pairs_with_keys)} (document, question) pairs")

    # Process in batches, checking cache before each batch
    all_triplets: list[dict[str, str]] = []
    total_processed = 0
    total_cached = 0

    # Batch size for processing (process this many pairs at a time)
    batch_size = cfg.cache_batch_size

    # Create oracle config (reused for all batches)
    oracle_config = OracleConfig(
        model_name=cfg.model_name,
        oracle_path=cfg.oracle_path,
        target_adapter_path=None,  # Use base model for initial dataset generation
        layer_percent=cfg.layer_percent,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        batch_size=8,  # Internal batch size for oracle evaluation
        device=cfg.device,
        dtype=cfg.dtype,
        load_in_8bit=cfg.load_in_8bit,
        token_start_idx=cfg.token_start_idx,
        token_end_idx=cfg.token_end_idx,
        repeats=cfg.repeats,
    )

    # Load oracle model once before processing batches (reused across all batches)
    logger.info("Loading oracle model for dataset generation...")
    oracle_model, oracle_tokenizer, oracle_dtype, oracle_device = _load_oracle_model(
        oracle_config
    )

    # Process in batches
    total_batches = (len(all_pairs_with_keys) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(all_pairs_with_keys), batch_size),
        desc="Processing batches",
        total=total_batches,
    ):
        batch_end = min(batch_start + batch_size, len(all_pairs_with_keys))
        batch_pairs_with_keys = all_pairs_with_keys[batch_start:batch_end]

        # Reload cache to pick up any changes (for resumability)
        cache = _load_cache(cache_dir)

        # Separate cached vs uncached pairs in this batch
        batch_cached_triplets: list[dict[str, str]] = []
        batch_uncached_pairs: list[tuple[str, str]] = []
        batch_uncached_keys: list[str] = []
        batch_uncached_indices: list[int] = []  # Original indices in batch

        for idx, (document, question, cache_key) in enumerate(batch_pairs_with_keys):
            if cache_key in cache:
                batch_cached_triplets.append(cache[cache_key])
                total_cached += 1
            else:
                batch_uncached_pairs.append((document, question))
                batch_uncached_keys.append(cache_key)
                batch_uncached_indices.append(idx)

        # Add cached triplets from this batch
        all_triplets.extend(batch_cached_triplets)

        # Process uncached pairs if any
        if batch_uncached_pairs:
            logger.info(
                f"Processing batch {batch_start // batch_size + 1}/{(len(all_pairs_with_keys) + batch_size - 1) // batch_size}: "
                f"{len(batch_uncached_pairs)} uncached pairs (batch {batch_start + 1}-{batch_end} of {len(all_pairs_with_keys)})"
            )

            # Run oracle on this batch (without judge scoring)
            # Pass pre-loaded model to avoid reloading for each batch
            oracle_results = run_oracle_eval_no_judge(
                oracle_config,
                batch_uncached_pairs,
                model=oracle_model,
                tokenizer=oracle_tokenizer,
                dtype=oracle_dtype,
                device=oracle_device,
                cleanup_adapters=False,  # Don't cleanup adapters when reusing model
            )

            # Save results to cache and add to all_triplets
            for i, (document, question) in enumerate(batch_uncached_pairs):
                oracle_result = oracle_results.results[i]
                oracle_answer = oracle_result.oracle_response
                cache_key = batch_uncached_keys[i]

                triplet = {
                    "document": document,
                    "question": question,
                    "oracle_answer": oracle_answer,
                }

                # Add to cache
                cache[cache_key] = triplet
                all_triplets.append(triplet)
                total_processed += 1

            # Save cache after processing this batch
            _save_cache(cache_dir, cache)
            logger.info(
                f"Saved batch {batch_start // batch_size + 1}: "
                f"{len(batch_uncached_pairs)} new triplets cached "
                f"(total: {total_processed} processed, {total_cached} from cache)"
            )
        else:
            logger.info(
                f"Batch {batch_start // batch_size + 1}: "
                f"All {len(batch_cached_triplets)} pairs already cached"
            )

    # Cleanup oracle adapter after all batches are processed
    if ORACLE_ADAPTER_NAME in oracle_model.peft_config:
        oracle_model.delete_adapter(ORACLE_ADAPTER_NAME)
        logger.info("Cleaned up oracle adapter")

    logger.info(
        f"Dataset generation complete: {total_processed} new triplets, "
        f"{total_cached} from cache, {len(all_triplets)} total"
    )

    # Build adversarial dataset with tokenization
    adversarial_dataset = []
    for triplet in all_triplets:
        document = triplet["document"]
        question = triplet["question"]
        oracle_answer = triplet["oracle_answer"]

        # Re-tokenize the document for training
        tokenized = tokenizer(
            document,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors=None,
        )

        adversarial_dataset.append(
            {
                "document": document,
                "question": question,
                "oracle_answer": oracle_answer,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        )

    logger.info(f"Generated {len(adversarial_dataset)} adversarial training examples")
    return adversarial_dataset


class AdversarialTrainer(Trainer):
    """Custom Trainer that computes adversarial loss in addition to SFT loss."""

    def __init__(
        self,
        adversarial_cfg: AdversarialExperimentConfig,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.adversarial_cfg = adversarial_cfg

        # Get layer for activation extraction
        self.activation_layer = layer_percent_to_layer(
            adversarial_cfg.model_name, adversarial_cfg.layer_percent
        )
        # Cast model to AutoModelForCausalLM for get_hf_submodule
        base_model = (
            self.model.base_model if hasattr(self.model, "base_model") else self.model
        )
        # Fix for Qwen3 models: nl_probes expects model.model.layers but Qwen3 has model.model.model.layers
        is_qwen3 = (
            "qwen" in adversarial_cfg.model_name.lower()
            and "3" in adversarial_cfg.model_name.lower()
        )
        if is_qwen3 and hasattr(base_model, "model"):
            peft_model_model = base_model.model
            if not hasattr(peft_model_model, "layers") and hasattr(
                peft_model_model, "model"
            ):
                inner_model = peft_model_model.model
                if hasattr(inner_model, "layers"):
                    peft_model_model.layers = inner_model.layers  # type: ignore[attr-defined]
        self.activation_submodule = get_hf_submodule(
            cast("AutoModelForCausalLM", base_model), self.activation_layer
        )

        # Get oracle injection layer (same as activation layer)
        # Use self.model (which has both adapters) with use_lora=True to get the oracle adapter's submodule
        # Fix for Qwen3 models: apply same fix to model
        if (
            is_qwen3
            and hasattr(self.model, "base_model")
            and hasattr(self.model.base_model, "model")
        ):
            qwen3_model = self.model.base_model.model
            if not hasattr(qwen3_model, "model") and hasattr(qwen3_model, "layers"):
                qwen3_model.model = qwen3_model  # type: ignore[attr-defined]
        # Pass self.model (PeftModel) when use_lora=True to access oracle adapter
        self.oracle_injection_submodule = get_hf_submodule(
            cast("AutoModelForCausalLM", self.model),
            self.activation_layer,
            use_lora=True,
        )

    def compute_loss(
        self,
        model: Any,  # nn.Module in base class, but we use PeftModel
        inputs: dict[str, t.Tensor | list[str] | Any],
        return_outputs: bool = False,
        num_items_in_batch: t.Tensor | None = None,  # noqa: ARG002
    ) -> Any:  # Return type matches base class (no annotation)
        """Compute combined SFT + adversarial loss.

        Flow:
        1. Forward pass on document to get SFT loss
        2. Collect activations at specified token positions (with gradients)
        3. Compute adversarial loss using collected activations
        4. Return combined loss: sft_loss + alpha * (-oracle_loss)
        """
        # Extract metadata
        questions = inputs.pop("question", [])
        oracle_answers = inputs.pop("oracle_answer", [])

        # Collect activations during forward pass
        activations_list: list[t.Tensor] = []

        def activation_hook(_module, _input, output):
            activations = output[0] if isinstance(output, tuple) else output
            activations_list.append(activations)

        activation_handle = self.activation_submodule.register_forward_hook(
            activation_hook
        )

        try:
            # Forward pass on document to get SFT loss and collect activations
            outputs = model(**inputs)
            sft_loss = outputs.loss
        finally:
            activation_handle.remove()

        # Extract collected activations
        assert activations_list, "Failed to collect activations during forward pass"
        activations_BTD = activations_list[0]  # [B, T, D]

        assert (
            activations_BTD.shape[0] == len(questions)
        ), f"Mismatch: batch_size={activations_BTD.shape[0]}, questions={len(questions)}"

        # Restore metadata for adversarial loss computation
        inputs["question"] = questions
        inputs["oracle_answer"] = oracle_answers

        # Compute adversarial loss using collected activations
        # Cast model to PeftModel since we know it's a PeftModel in this context
        peft_model = cast("PeftModel", model)
        adv_loss = self._compute_adversarial_loss(peft_model, inputs, activations_BTD)

        # Combined loss: SFT loss + alpha * (negative oracle loss)
        total_loss = sft_loss + self.adversarial_cfg.adversarial_alpha * adv_loss

        if return_outputs:
            return total_loss, outputs

        return total_loss

    def create_optimizer(self) -> None:
        """Create optimizer, including only trainable LoRA adapter parameters.

        Oracle adapter parameters must have requires_grad=True for gradient flow,
        but we exclude them from the optimizer so they don't get updated.
        Base model parameters are also excluded (even if requires_grad=True),
        as we're only training the LoRA adapter.
        """
        # Get all parameters that should be optimized (only trainable LoRA adapter)
        decay_parameters = []
        no_decay_parameters = []
        if self.model is None:
            raise ValueError("Model is None, cannot create optimizer")
        for name, param in self.model.named_parameters():
            # Skip oracle adapter parameters
            if ORACLE_ADAPTER_NAME in name:
                continue

            # Only include parameters that belong to our trainable adapter
            # PEFT names adapter parameters with the adapter name in the path
            if TRAINABLE_ADAPTER_NAME not in name:
                continue

            # Only include parameters that require gradients
            if not param.requires_grad:
                continue

            # Group parameters for weight decay
            if len(param.shape) >= 2 and "bias" not in name:
                decay_parameters.append(param)
            else:
                no_decay_parameters.append(param)

        # Create optimizer groups
        optimizer_grouped_parameters = [
            {
                "params": decay_parameters,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ]

        # Create optimizer using the same logic as Trainer
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # Create learning rate scheduler
        if self.args.max_steps > 0:
            num_training_steps = self.args.max_steps
        else:
            num_training_steps = (
                len(self.get_train_dataloader())
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Ensure num_training_steps is an int
        num_training_steps_int = int(num_training_steps)
        warmup_steps = self.args.get_warmup_steps(num_training_steps_int)

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps_int,
        )

    def _compute_adversarial_loss(
        self,
        model: PeftModel,
        inputs: dict[str, t.Tensor | list[str]],
        activations_BTD: t.Tensor,
    ) -> t.Tensor:
        """Compute adversarial loss by maximizing oracle error on answer tokens.

        Flow:
        1. Extract activations from specified token positions
        2. Disable our LoRA, enable oracle LoRA
        3. Format question + answer with ACT tokens at the beginning
        4. Inject activations at ACT token positions using steering hook
        5. Forward pass through oracle
        6. Compute negative cross-entropy loss on answer tokens only
        """
        batch_size, seq_len, _d_model = activations_BTD.shape
        device = inputs["input_ids"].device  # type: ignore

        # Get questions and answers - must be present
        questions_raw = inputs.pop("question")
        answers_raw = inputs.pop("oracle_answer")

        # Convert to lists of strings
        questions = (
            [str(q) for q in questions_raw]
            if isinstance(questions_raw, list)
            else [str(questions_raw)]
        )
        answers = (
            [str(a) for a in answers_raw]
            if isinstance(answers_raw, list)
            else [str(answers_raw)]
        )

        assert (
            len(questions) == batch_size and len(answers) == batch_size
        ), f"Mismatch: batch_size={batch_size}, questions={len(questions)}, answers={len(answers)}"

        # Extract activations from specified token positions
        token_start = seq_len + self.adversarial_cfg.token_start_idx
        token_end = seq_len + self.adversarial_cfg.token_end_idx

        assert (
            0 <= token_start < token_end <= seq_len
        ), f"Invalid token indices: start={token_start}, end={token_end}, seq_length={seq_len}"

        # Switch to oracle adapter (for adversarial loss computation)
        # Keep model in training mode to allow gradient flow
        model.set_adapter(ORACLE_ADAPTER_NAME)

        try:
            adv_losses = []

            for batch_idx in range(batch_size):
                # Extract activations for this sample
                segment_activations_KD = activations_BTD[
                    batch_idx, token_start:token_end, :
                ]  # [K, D]
                question = questions[batch_idx]
                answer = answers[batch_idx]

                # Create oracle input with ACT tokens
                # The oracle expects: ACT tokens + question + answer
                # ACT tokens are created by get_introspection_prefix
                num_positions = (
                    segment_activations_KD.shape[0] * self.adversarial_cfg.repeats
                )
                act_prefix = get_introspection_prefix(
                    self.activation_layer, num_positions
                )
                prompt = act_prefix + question

                # Format question as chat message
                if self.processing_class is None:
                    raise ValueError("Tokenizer is None")
                # Type assertion: processing_class is a PreTrainedTokenizer
                tokenizer = cast("PreTrainedTokenizer", self.processing_class)
                prompt_messages: list[dict[str, str]] = [
                    {"role": "user", "content": prompt}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                assert isinstance(prompt_text, str), "Prompt text must be a string"

                # Tokenize
                prompt_inputs_dict = tokenizer(
                    prompt_text,
                    padding=True,
                    truncation=True,
                    max_length=self.adversarial_cfg.max_length,
                )
                answer_inputs_dict = tokenizer(
                    answer,
                    padding=True,
                    truncation=True,
                    max_length=self.adversarial_cfg.max_length,
                )
                full_inputs_dict = {
                    key: prompt_inputs_dict[key] + answer_inputs_dict[key]
                    for key in prompt_inputs_dict
                }

                # Answer starts after ACT prefix + question
                answer_start = len(prompt_inputs_dict["input_ids"])
                full_seq_len = len(full_inputs_dict["input_ids"])

                assert (
                    answer_start < full_seq_len
                ), f"Answer start position {answer_start} >= sequence length {full_seq_len}"

                # Prepare activations for injection at ACT token positions
                # Repeat activations according to config
                repeated_activations = segment_activations_KD.repeat(
                    self.adversarial_cfg.repeats, 1
                )  # [K*repeats, D]
                num_act_tokens = repeated_activations.shape[0]

                act_token_positions = find_pattern_in_tokens(
                    full_inputs_dict["input_ids"],
                    SPECIAL_TOKEN,
                    num_act_tokens,
                    cast("AutoTokenizer", tokenizer),
                )

                assert (
                    len(act_token_positions) == num_act_tokens
                ), f"Expected {num_act_tokens} ACT token positions, got {len(act_token_positions)}"

                # Create steering hook
                steering_hook = get_hf_activation_steering_hook(
                    vectors=[
                        repeated_activations
                    ],  # List of [K, D] tensors, one per batch item
                    positions=[
                        act_token_positions
                    ],  # List of position lists, one per batch item
                    steering_coefficient=1.0,
                    device=device,
                    dtype=segment_activations_KD.dtype,
                )

                full_inputs_dict = {
                    key: t.tensor(value, device=device).unsqueeze(0)
                    for key, value in full_inputs_dict.items()
                }
                labels = full_inputs_dict["input_ids"].clone()
                labels[:, :answer_start] = -100
                full_inputs_dict["labels"] = labels

                # Forward pass through oracle with steering
                if self.model is None:
                    raise ValueError("Model is None")
                with (
                    add_hook(self.oracle_injection_submodule, steering_hook),
                    t.enable_grad(),
                ):
                    oracle_outputs = self.model(**full_inputs_dict)
                    loss = oracle_outputs.loss

                    # Negate to maximize (gradient ascent on oracle loss)
                    negated_loss = -loss
                    adv_losses.append(negated_loss)

            stacked_loss = t.stack(adv_losses)
            mean_loss = stacked_loss.mean()
            return mean_loss

        finally:
            # Switch back to our trainable adapter
            model.set_adapter(TRAINABLE_ADAPTER_NAME)


def config_to_experiment_config(cfg: DictConfig) -> AdversarialExperimentConfig:
    """Convert Hydra DictConfig to AdversarialExperimentConfig dataclass."""
    return AdversarialExperimentConfig(
        model_name=cfg.model.name,
        tokenizer_name=cfg.model.tokenizer,
        lora_enabled=cfg.model.lora.enabled,
        lora_r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        lora_target_modules=list(cfg.model.lora.target_modules),
        dataset_name=cfg.data.name,
        max_length=cfg.data.max_length,
        train_ratio=cfg.data.split.train_ratio,
        max_samples=cfg.data.max_samples,
        training_subset_size=cfg.data.get("training_subset_size", None),
        max_messages_per_conversation=cfg.data.max_messages_per_conversation,
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        adversarial_alpha=cfg.training.adversarial.alpha,
        oracle_path=cfg.training.adversarial.oracle_path,
        layer_percent=cfg.training.adversarial.layer_percent,
        token_start_idx=cfg.training.adversarial.token_start_idx,
        token_end_idx=cfg.training.adversarial.token_end_idx,
        repeats=cfg.training.adversarial.repeats,
        question_seed=cfg.experiment.seed,
        dtype=cfg.hardware.dtype,
        device=cfg.hardware.device,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        attn_implementation=cfg.model.attn_implementation,
        seed=cfg.experiment.seed,
        output_dir=cfg.experiment.output_dir,
        cache_dir=cfg.experiment.get("cache_dir", ".cache/adv-thought"),
        cache_batch_size=cfg.experiment.get("cache_batch_size", 32),
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Suppress transformers warnings that interfere with tqdm progress bar
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    logger.info("Starting adversarial SFT finetuning experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Convert to typed config
    exp_cfg = config_to_experiment_config(cfg)

    # Set seed for reproducibility
    set_seed(exp_cfg.seed)

    # Initialize W&B if enabled
    if exp_cfg.wandb_enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=exp_cfg.wandb_project,
            name=cfg.experiment.name,
            config=cast("dict[str, Any]", config_dict)
            if isinstance(config_dict, dict)
            else None,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        )

    # Load base model and tokenizer (without LoRA yet)
    logger.info(f"Loading base model: {exp_cfg.model_name}")
    dtype = get_dtype(exp_cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        exp_cfg.tokenizer_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up quantization config if needed
    quantization_config = None
    if exp_cfg.load_in_4bit:
        logger.info("Loading model with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif exp_cfg.load_in_8bit:
        logger.info("Loading model with 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    # Load base model
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto" if exp_cfg.device == "cuda" else None,
        "trust_remote_code": True,
        "attn_implementation": exp_cfg.attn_implementation,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForCausalLM.from_pretrained(
        exp_cfg.model_name,
        **model_kwargs,
    )

    # Load oracle adapter first (will be frozen)
    logger.info(f"Loading oracle adapter: {exp_cfg.oracle_path}")
    model = PeftModel.from_pretrained(
        base_model,
        exp_cfg.oracle_path,
        adapter_name=ORACLE_ADAPTER_NAME,
    )

    # Add our trainable adapter
    if exp_cfg.lora_enabled:
        logger.info("Adding trainable LoRA adapter...")
        lora_config = LoraConfig(
            r=exp_cfg.lora_r,
            lora_alpha=exp_cfg.lora_alpha,
            lora_dropout=exp_cfg.lora_dropout,
            target_modules=exp_cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.add_adapter(TRAINABLE_ADAPTER_NAME, lora_config)
        model.set_adapter(TRAINABLE_ADAPTER_NAME)  # Set our adapter as active
        model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if exp_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # When using k-bit quantization + gradient checkpointing, ensure at least one
    # input to checkpointed blocks requires grad; otherwise losses can become
    # non-differentiable (grad_fn=None) and backward() will fail.
    if (
        exp_cfg.gradient_checkpointing
        and (exp_cfg.load_in_8bit or exp_cfg.load_in_4bit)
        and hasattr(model, "enable_input_require_grads")
    ):
        model.enable_input_require_grads()

    # Load and prepare dataset
    datasets = load_and_prepare_conversation_dataset(
        dataset_name=exp_cfg.dataset_name,
        tokenizer=tokenizer,
        seed=exp_cfg.seed,
        train_ratio=exp_cfg.train_ratio,
        max_length=exp_cfg.max_length,
        max_samples=exp_cfg.max_samples,
        max_messages_per_conversation=exp_cfg.max_messages_per_conversation,
    )

    # Generate adversarial dataset
    adversarial_dataset = generate_adversarial_dataset(datasets, tokenizer, exp_cfg)

    # Convert to HuggingFace Dataset format
    adversarial_hf_dataset = Dataset.from_list(adversarial_dataset)

    # Apply training subset size limit if specified
    if exp_cfg.training_subset_size is not None:
        logger.info(
            f"Limiting adversarial training dataset to first {exp_cfg.training_subset_size} samples"
        )
        adversarial_hf_dataset = adversarial_hf_dataset.select(
            range(min(exp_cfg.training_subset_size, len(adversarial_hf_dataset)))
        )
        logger.info(
            f"Adversarial dataset size after subset selection: {len(adversarial_hf_dataset)}"
        )

    # Split into train/val (use same split ratio as original)
    split_result = adversarial_hf_dataset.train_test_split(
        test_size=1 - exp_cfg.train_ratio, seed=exp_cfg.seed
    )
    adversarial_datasets = {
        "train": split_result["train"],
        "validation": split_result["test"],
    }

    # Create custom data collator that preserves question and oracle_answer fields
    class AdversarialDataCollator(DataCollatorForLanguageModeling):
        def __call__(
            self, features: list[dict[str, Any]], return_tensors: str | None = None
        ) -> dict[str, Any]:
            # Extract metadata
            questions = [f["question"] for f in features]
            oracle_answers = [f["oracle_answer"] for f in features]

            # Remove metadata from features before collation
            collate_features = []
            for f in features:
                collate_f = {
                    k: v
                    for k, v in f.items()
                    if k not in ["question", "oracle_answer", "document"]
                }
                collate_features.append(collate_f)

            # Collate as normal
            batch = super().__call__(collate_features, return_tensors=return_tensors)

            # Add metadata back
            batch["question"] = questions
            batch["oracle_answer"] = oracle_answers

            return batch

    data_collator = AdversarialDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=exp_cfg.output_dir,
        num_train_epochs=exp_cfg.num_epochs,
        per_device_train_batch_size=exp_cfg.batch_size,
        per_device_eval_batch_size=exp_cfg.batch_size,
        gradient_accumulation_steps=exp_cfg.gradient_accumulation_steps,
        learning_rate=exp_cfg.learning_rate,
        weight_decay=exp_cfg.weight_decay,
        warmup_ratio=exp_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=exp_cfg.max_grad_norm,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=exp_cfg.dtype == "bfloat16",
        fp16=exp_cfg.dtype == "float16",
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to="wandb" if exp_cfg.wandb_enabled else "none",
        seed=exp_cfg.seed,
    )

    # Create trainer
    trainer = AdversarialTrainer(
        adversarial_cfg=exp_cfg,
        model=model,
        args=training_args,
        train_dataset=adversarial_datasets["train"],
        eval_dataset=adversarial_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Fix a bug with the question and oracle_answer columns being removed
    trainer._set_signature_columns_if_needed()
    if trainer._signature_columns is not None:
        trainer._signature_columns += ["question", "oracle_answer"]
    else:
        trainer._signature_columns = ["question", "oracle_answer"]

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {exp_cfg.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(exp_cfg.output_dir)

    # Save trainable adapter to root directory for evaluation compatibility
    # PEFT's from_pretrained expects adapter_config.json at root, not in subdirectories
    if exp_cfg.lora_enabled:
        logger.info("Saving trainable adapter to root directory for evaluation...")
        output_path = Path(exp_cfg.output_dir)
        trainable_dir = output_path / TRAINABLE_ADAPTER_NAME
        if trainable_dir.exists():
            # Copy adapter_config.json to root
            adapter_config_src = trainable_dir / "adapter_config.json"
            if adapter_config_src.exists():
                shutil.copy2(adapter_config_src, output_path / "adapter_config.json")
                logger.info("Copied adapter_config.json to root")
            # Copy adapter_model files (could be .safetensors or .bin)
            adapter_model_files = list(trainable_dir.glob("adapter_model.*"))
            if adapter_model_files:
                for adapter_file in adapter_model_files:
                    shutil.copy2(adapter_file, output_path / adapter_file.name)
                    logger.info(f"Copied {adapter_file.name} to root")
            else:
                logger.warning("No adapter_model files found in trainable directory")
        else:
            logger.warning(f"Trainable adapter directory not found: {trainable_dir}")

    # Finish W&B run
    if exp_cfg.wandb_enabled:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
