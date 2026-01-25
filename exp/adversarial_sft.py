"""Adversarial Supervised Fine-Tuning (SFT) experiment.

This module implements adversarial SFT training where we:
1. Generate a dataset of (document, question, oracle_answer) triples
2. During training, compute both SFT loss and adversarial loss
3. Adversarial loss maximizes oracle error on answer tokens
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch as t
from loguru import logger

# Import nl_probes utilities
from nl_probes.base_experiment import (
    load_model,
    load_tokenizer,
)
from nl_probes.utils.activation_utils import (
    get_hf_submodule,
)
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import create_training_datapoint
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import hydra
import wandb
from core.data import load_and_prepare_conversation_dataset
from core.dtype import get_dtype
from core.questions import get_train_questions
from core.steering_hooks import add_hook, get_hf_activation_steering_hook
from exp.oracle import OracleConfig, run_oracle_eval


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

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "adversarial-thought"


ORACLE_ADAPTER_NAME = "oracle"


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


def generate_adversarial_dataset(
    datasets: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cfg: AdversarialExperimentConfig,
) -> list[dict[str, Any]]:
    """Generate adversarial dataset with (document, question, oracle_answer) triples.

    Args:
        datasets: DatasetDict with train/validation splits
        tokenizer: Tokenizer for formatting
        cfg: Experiment configuration

    Returns:
        List of dicts with keys: document, question, oracle_answer, input_ids, attention_mask
    """
    logger.info("Generating adversarial dataset...")

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

    # Prepare context-question pairs
    context_question_pairs: list[tuple[str, str]] = []
    documents: list[str] = []

    for example in train_dataset:
        # Extract document text from tokenized example
        # The dataset is already tokenized, so we need to decode it
        input_ids = example["input_ids"]
        document_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        if len(document_text.strip()) == 0:
            continue

        # Randomly select one question per document
        question = rng.choice(train_questions)
        context_question_pairs.append((document_text, question))
        documents.append(document_text)

    logger.info(f"Generated {len(context_question_pairs)} (document, question) pairs")

    # Run oracle to get answers
    oracle_config = OracleConfig(
        model_name=cfg.model_name,
        oracle_path=cfg.oracle_path,
        target_adapter_path=None,  # Use base model for initial dataset generation
        layer_percent=cfg.layer_percent,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        batch_size=8,
        device=cfg.device,
        dtype=cfg.dtype,
        load_in_8bit=cfg.load_in_8bit,
        token_start_idx=cfg.token_start_idx,
        token_end_idx=cfg.token_end_idx,
        repeats=cfg.repeats,
    )

    logger.info("Running oracle to get answers...")
    oracle_results = run_oracle_eval(oracle_config, context_question_pairs)

    # Build adversarial dataset
    adversarial_dataset = []
    for i, (document, question) in enumerate(context_question_pairs):
        oracle_result = oracle_results.results[i]
        oracle_answer = oracle_result.oracle_response

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
        oracle_model: PeftModel,
        oracle_tokenizer: PreTrainedTokenizer,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.adversarial_cfg = adversarial_cfg
        self.oracle_model = oracle_model
        self.oracle_tokenizer = oracle_tokenizer

        # Get layer for activation extraction
        self.activation_layer = layer_percent_to_layer(
            adversarial_cfg.model_name, adversarial_cfg.layer_percent
        )
        # Cast model to AutoModelForCausalLM for get_hf_submodule
        base_model = (
            self.model.base_model if hasattr(self.model, "base_model") else self.model
        )
        self.activation_submodule = get_hf_submodule(
            cast("AutoModelForCausalLM", base_model), self.activation_layer
        )

        # Get oracle injection layer (same as activation layer)
        # Cast oracle_model to AutoModelForCausalLM for get_hf_submodule
        oracle_base_model = (
            oracle_model.base_model
            if hasattr(oracle_model, "base_model")
            else oracle_model
        )
        self.oracle_injection_submodule = get_hf_submodule(
            cast("AutoModelForCausalLM", oracle_base_model),
            self.activation_layer,
            use_lora=True,
        )

    def compute_loss(
        self,
        model: t.nn.Module,
        inputs: dict[str, t.Tensor | list[str] | Any],
        return_outputs: bool = False,
        num_items_in_batch: t.Tensor | None = None,  # noqa: ARG002
    ) -> t.Tensor | tuple[t.Tensor, dict[str, Any]]:
        """Compute combined SFT + adversarial loss."""
        # Extract metadata before passing to model
        questions = inputs.pop("question", [])
        oracle_answers = inputs.pop("oracle_answer", [])

        # Standard SFT loss (on document only)
        outputs = model(**inputs)
        sft_loss = outputs.loss

        # Compute adversarial loss
        # Restore metadata for adversarial loss computation
        inputs["question"] = questions
        inputs["oracle_answer"] = oracle_answers
        adv_loss = self._compute_adversarial_loss(model, inputs)

        # Combined loss
        total_loss = sft_loss + self.adversarial_cfg.adversarial_alpha * adv_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_adversarial_loss(
        self,
        model: t.nn.Module,
        inputs: dict[str, t.Tensor | list[str]],
    ) -> t.Tensor:
        """Compute adversarial loss by maximizing oracle error on answer tokens.

        Flow:
        1. Collect activations from tokens -10 to -2 during forward pass (with gradients)
        2. Switch to oracle LoRA
        3. Create verbalizer inputs from activations + question
        4. Forward pass through oracle with question + answer
        5. Compute negative cross-entropy loss on answer tokens
        """
        batch_size = inputs["input_ids"].shape[0]  # type: ignore
        device = inputs["input_ids"].device  # type: ignore

        # Get questions and answers from the batch
        if "question" not in inputs or "oracle_answer" not in inputs:
            logger.warning(
                "Questions/answers not found in inputs, skipping adversarial loss"
            )
            return t.tensor(0.0, device=device)

        questions_raw = inputs["question"]
        answers_raw = inputs["oracle_answer"]

        # Ensure they are lists of strings
        if isinstance(questions_raw, list):
            questions = [str(q) for q in questions_raw]
        else:
            questions = [str(questions_raw)]

        if isinstance(answers_raw, list):
            answers = [str(a) for a in answers_raw]
        else:
            answers = [str(answers_raw)]

        # Ensure we have enough items
        if len(questions) < batch_size or len(answers) < batch_size:
            logger.warning(
                f"Insufficient questions/answers: {len(questions)}/{len(answers)} for batch_size {batch_size}"
            )
            return t.tensor(0.0, device=device)

        # Collect activations from the forward pass (with gradients enabled for training LoRA)
        # We need to hook into the forward pass to capture activations
        # Note: We need gradients to flow through activations for adversarial training
        activations_list = []

        def activation_hook(_module, _input, output):
            # Extract activations from output
            # Keep gradients enabled (don't detach) so we can backprop through them
            activations = output[0] if isinstance(output, tuple) else output
            activations_list.append(activations)

        handle = self.activation_submodule.register_forward_hook(activation_hook)

        try:
            # Forward pass to collect activations (gradients enabled for training LoRA)
            # We already did a forward pass in compute_loss, so we need to do another one
            # or reuse the outputs. Let's do another forward pass to get activations.
            model(**inputs)
        finally:
            handle.remove()

        if not activations_list:
            return t.tensor(0.0, device=device)

        activations_BLD = activations_list[0]  # [B, L, D]

        # Extract activations from tokens -10 to -2 for each sample
        seq_length = activations_BLD.shape[1]
        token_start = seq_length + self.adversarial_cfg.token_start_idx
        token_end = seq_length + self.adversarial_cfg.token_end_idx

        # Ensure valid indices
        token_start = max(0, token_start)
        token_end = min(seq_length, token_end)

        if token_start >= token_end:
            # Fallback: use last few tokens
            token_start = max(0, seq_length - 10)
            token_end = seq_length

        # Compute adversarial loss for each sample
        adv_losses = []

        for b in range(batch_size):
            # Get activations for this sample's segment
            segment_activations_KD = activations_BLD[
                b, token_start:token_end, :
            ]  # [K, D]
            question = questions[b]
            answer = answers[b]

            # Create verbalizer input from activations + question
            # We need to create TrainingDataPoint and feed through oracle
            # For segment input type, we repeat the activations
            segment_repeats = self.adversarial_cfg.repeats

            # Create training datapoints for the oracle
            verbalizer_inputs = []
            for repeat_idx in range(segment_repeats):
                # Cast tokenizer to AutoTokenizer for create_training_datapoint
                oracle_auto_tokenizer = cast("AutoTokenizer", self.oracle_tokenizer)
                dp = create_training_datapoint(
                    datapoint_type="segment",
                    prompt=str(question),
                    target_response=str(answer),
                    layer=self.activation_layer,
                    num_positions=segment_activations_KD.shape[0],
                    tokenizer=oracle_auto_tokenizer,
                    acts_BD=segment_activations_KD,  # [K, D]
                    feature_idx=-1,
                    context_input_ids=None,
                    context_positions=None,
                    ds_label=None,
                    meta_info={"repeat": repeat_idx},
                )
                verbalizer_inputs.append(dp)

            # Switch to oracle adapter
            was_training = self.oracle_model.training
            self.oracle_model.eval()
            self.oracle_model.set_adapter(ORACLE_ADAPTER_NAME)

            try:
                # Create verbalizer input: question + answer
                # Format question as chat message
                question_str = str(question)
                answer_str = str(answer)
                question_messages: list[dict[str, str]] = [
                    {"role": "user", "content": question_str}
                ]
                question_text = self.oracle_tokenizer.apply_chat_template(
                    question_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Full sequence: question + answer
                if isinstance(question_text, str):
                    full_text = question_text + answer_str
                else:
                    # If apply_chat_template returns something else, convert to string
                    full_text = str(question_text) + answer_str

                # Tokenize the full sequence
                oracle_inputs_dict = self.oracle_tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                oracle_inputs = {k: v.to(device) for k, v in oracle_inputs_dict.items()}

                # Get token positions for question vs answer
                # Tokenize question with chat template to get exact tokenization
                question_tokens_full = self.oracle_tokenizer(
                    question_text,
                    return_tensors="pt",
                    add_special_tokens=True,  # Chat template already includes special tokens
                )["input_ids"].to(device)

                # Tokenize answer separately
                answer_tokens = self.oracle_tokenizer(
                    answer,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(device)

                # The full sequence tokenization should match question_tokens_full + answer_tokens
                # But we need to account for any special tokens between them
                # For now, use the length of question_tokens_full as answer_start
                question_len = question_tokens_full.shape[1]
                answer_start = question_len
                answer_end = answer_start + answer_tokens.shape[1]

                # Verify the full sequence matches
                full_seq_len = oracle_inputs["input_ids"].shape[1]
                if answer_end > full_seq_len:
                    # Adjust if answer doesn't fit
                    answer_end = full_seq_len
                    logger.warning(
                        f"Answer truncated: expected {answer_start + answer_tokens.shape[1]}, got {full_seq_len}"
                    )

                # Prepare activations for steering
                # The steering hook expects:
                # - vectors: list of [K_b, D] tensors, one per batch item
                # - positions: list of lists of positions, one per batch item
                segment_len = segment_activations_KD.shape[0]  # K

                # For segment input, we inject at multiple positions (one per activation in segment)
                # We'll inject at positions starting from answer_start
                num_injections = min(
                    segment_len, 5
                )  # Limit to avoid too many injections
                injection_positions = []
                injection_vectors = []

                for i in range(num_injections):
                    pos = answer_start + i
                    if (
                        pos < oracle_inputs["input_ids"].shape[1]
                    ):  # Ensure valid position
                        injection_positions.append(pos)
                        injection_vectors.append(
                            segment_activations_KD[i]
                        )  # [D] tensor with gradients

                if not injection_vectors:
                    adv_losses.append(t.tensor(0.0, device=device))
                    continue

                # Stack vectors into [K, D] tensor for this batch item
                steering_vector_batch = t.stack(injection_vectors, dim=0)  # [K, D]

                # Create steering hook
                # For single sample: vectors = [steering_vector_batch], positions = [injection_positions]
                steering_hook = get_hf_activation_steering_hook(
                    vectors=[
                        steering_vector_batch
                    ],  # List of [K, D] tensors, one per batch
                    positions=[
                        injection_positions
                    ],  # List of lists of positions, one per batch
                    steering_coefficient=1.0,  # Full steering
                    device=device,
                    dtype=segment_activations_KD.dtype,
                )

                # Forward pass through oracle with steering
                # Enable gradients for this forward pass so we can backprop through activations
                with add_hook(self.oracle_injection_submodule, steering_hook):
                    oracle_outputs = self.oracle_model(**oracle_inputs)
                    logits = oracle_outputs.logits  # [1, L, V]

                # Create labels: -100 for question tokens, actual tokens for answer
                labels = oracle_inputs["input_ids"].clone()
                labels[0, :answer_start] = -100  # Ignore question tokens

                # Compute cross-entropy loss (only on answer tokens)
                shift_logits = logits[0, :-1, :].contiguous()  # [L-1, V]
                shift_labels = labels[0, 1:].contiguous()  # [L-1]

                loss_fn = t.nn.CrossEntropyLoss(ignore_index=-100)
                answer_loss = loss_fn(shift_logits, shift_labels)

                # Negate to maximize (we want high loss on answer = low probability)
                adv_losses.append(-answer_loss)

            finally:
                if was_training:
                    self.oracle_model.train()

        if len(adv_losses) == 0:
            return t.tensor(0.0, device=device)

        return t.stack(adv_losses).mean()


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
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
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

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(exp_cfg)

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
    from datasets import Dataset

    adversarial_hf_dataset = Dataset.from_list(adversarial_dataset)

    # Split into train/val (use same split ratio as original)
    split_result = adversarial_hf_dataset.train_test_split(
        test_size=1 - exp_cfg.train_ratio, seed=exp_cfg.seed
    )
    adversarial_datasets = {
        "train": split_result["train"],
        "validation": split_result["test"],
    }

    # Load oracle model
    logger.info("Loading oracle model...")
    dtype = get_dtype(exp_cfg.dtype)

    oracle_tokenizer = load_tokenizer(exp_cfg.model_name)
    oracle_base_model = load_model(exp_cfg.model_name, dtype)
    oracle_model = PeftModel.from_pretrained(
        oracle_base_model,  # type: ignore[arg-type]
        exp_cfg.oracle_path,
        adapter_name=ORACLE_ADAPTER_NAME,
    )
    oracle_model.eval()
    oracle_model.set_adapter(ORACLE_ADAPTER_NAME)

    # Freeze oracle model
    for param in oracle_model.parameters():
        param.requires_grad = False

    # Create custom data collator that preserves question and oracle_answer fields
    class AdversarialDataCollator(DataCollatorForLanguageModeling):
        def __call__(
            self, features: list[dict[str, Any]], return_tensors: str | None = None
        ) -> dict[str, Any]:
            # Extract metadata
            questions = [f.get("question", "") for f in features]
            oracle_answers = [f.get("oracle_answer", "") for f in features]

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
    # Cast oracle_tokenizer to PreTrainedTokenizer for AdversarialTrainer
    trainer = AdversarialTrainer(
        adversarial_cfg=exp_cfg,
        oracle_model=oracle_model,
        oracle_tokenizer=cast("PreTrainedTokenizer", oracle_tokenizer),
        model=model,
        args=training_args,
        train_dataset=adversarial_datasets["train"],
        eval_dataset=adversarial_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {exp_cfg.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(exp_cfg.output_dir)

    # Finish W&B run
    if exp_cfg.wandb_enabled:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
