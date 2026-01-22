from typing import Any

from datasets import DatasetDict, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


def load_and_prepare_conversation_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    max_length: int = 2048,
    max_samples: int | None = None,
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
    logger.info(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")

    # Limit samples if specified
    if max_samples is not None:
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset = dataset.shuffle(seed=seed)
    split_result = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    train_dataset = split_result["train"]
    val_dataset = split_result["test"]

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def tokenize_conversation(examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize conversations using the chat template."""
        texts = []

        # Map unsupported roles to supported ones
        role_map = {"environment": "user"}

        for messages in examples["messages"]:
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
