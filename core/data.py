from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def load_and_split_dataset(
    dataset_name: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    max_samples: int | None = None,
) -> DatasetDict:
    """Load a dataset, apply filtering, shuffle, and split into train/validation.

    Args:
        dataset_name: Name of the dataset to load (e.g., "allenai/Dolci-Instruct-SFT")
        seed: Random seed for shuffling and splitting
        train_ratio: Ratio of data to use for training (rest goes to validation)
        max_samples: Maximum number of samples to use from the dataset. If None, uses all samples.

    Returns:
        DatasetDict with "train" and "validation" splits
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load the dataset - using split="train" returns a Dataset (not DatasetDict)
    dataset = load_dataset(dataset_name, split="train")
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"
    assert len(dataset) > 0, "Dataset must not be empty"

    # Limit samples if specified
    if max_samples is not None:
        logger.info(f"Limiting dataset to {max_samples} samples")
        assert max_samples > 0, f"max_samples must be positive, got {max_samples}"
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        assert len(dataset) > 0, "Dataset must not be empty after filtering"

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    split_result = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_dataset = split_result["train"]
    val_dataset = split_result["test"]

    assert isinstance(train_dataset, Dataset), "Train dataset must be a Dataset"
    assert isinstance(val_dataset, Dataset), "Validation dataset must be a Dataset"
    assert len(train_dataset) > 0, "Train dataset must not be empty"
    assert len(val_dataset) > 0, "Validation dataset must not be empty"

    logger.info(
        f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation"
    )

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def load_and_prepare_conversation_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    max_length: int = 2048,
    max_samples: int | None = None,
    max_messages_per_conversation: int = 3,
) -> DatasetDict:
    """Load a conversation dataset, split it, and tokenize it for training.

    Args:
        dataset_name: Name of the dataset to load (e.g., "allenai/Dolci-Instruct-SFT")
        tokenizer: Tokenizer to use for tokenization
        seed: Random seed for shuffling
        train_ratio: Ratio of data to use for training (rest goes to validation)
        max_length: Maximum sequence length for tokenization
        max_samples: Maximum number of samples to use from the dataset. If None, uses all samples.

    Returns:
        DatasetDict with "train" and "validation" splits, tokenized and ready for training
    """
    # Load and split dataset using shared helper
    datasets = load_and_split_dataset(
        dataset_name=dataset_name,
        seed=seed,
        train_ratio=train_ratio,
        max_samples=max_samples,
    )
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    def tokenize_conversation(examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize conversations using the chat template."""
        texts = []

        # Map unsupported roles to supported ones
        role_map = {"environment": "user"}

        for messages in examples["messages"]:
            # Truncate each conversation to at most the configured number of messages
            messages = messages[:max_messages_per_conversation]
            # Clean messages: keep only role and content, map unsupported roles
            # Also merge consecutive messages with the same role
            cleaned_messages: list[dict[str, str]] = []
            for msg in messages:
                role = msg.get("role", "user")
                role = role_map.get(role, role)  # Map unsupported roles
                content = msg.get("content", "")
                if not content:  # Skip empty messages
                    continue
                # Merge consecutive messages with the same role
                if cleaned_messages and cleaned_messages[-1]["role"] == role:
                    cleaned_messages[-1]["content"] += "\n\n" + content
                else:
                    cleaned_messages.append({"role": role, "content": content})

            # Ensure conversation starts properly (user first, after optional system)
            # Remove leading assistant messages
            while cleaned_messages and cleaned_messages[0]["role"] == "assistant":
                cleaned_messages.pop(0)

            if not cleaned_messages:
                # Skip empty conversations
                texts.append("")
                continue

            # Apply chat template to format the conversation
            text = tokenizer.apply_chat_template(
                cleaned_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(str(text))

        # Tokenize all texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        return tokenized

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        tokenize_conversation,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_dataset.map(
        tokenize_conversation,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation",
    )

    return DatasetDict({"train": train_tokenized, "validation": val_tokenized})
