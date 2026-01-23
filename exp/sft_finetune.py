"""Supervised Fine-Tuning (SFT) experiment for Gemma-3-1B on Dolci-Instruct-SFT."""

import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import hydra
import wandb
from core.data import load_and_prepare_conversation_dataset
from core.dtype import get_dtype


@dataclass
class ExperimentConfig:
    """Flattened experiment configuration for type safety."""

    # Model
    model_name: str = "google/gemma-3-1b-it"
    tokenizer_name: str = "google/gemma-3-1b-it"

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

    # Hardware
    dtype: str = "bfloat16"
    device: str = "cuda"

    # Experiment
    seed: int = 42
    output_dir: str = "out/sft_baseline"

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "adversarial-thought"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    cfg: ExperimentConfig,
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if cfg.device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
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


def create_training_arguments(cfg: ExperimentConfig) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=cfg.dtype == "bfloat16",
        fp16=cfg.dtype == "float16",
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to="wandb" if cfg.wandb_enabled else "none",
        seed=cfg.seed,
    )


def config_to_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra DictConfig to ExperimentConfig dataclass."""
    return ExperimentConfig(
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
        dtype=cfg.hardware.dtype,
        device=cfg.hardware.device,
        seed=cfg.experiment.seed,
        output_dir=cfg.experiment.output_dir,
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting SFT finetuning experiment")
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

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create training arguments
    training_args = create_training_arguments(exp_cfg)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
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
