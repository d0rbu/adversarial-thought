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
) -> DatasetDict:
    """Load a conversation dataset, split it, and tokenize it for training.

    Args:
        dataset_name: Name of the dataset to load (e.g., "allenai/Dolci-Instruct-SFT")
        tokenizer: Tokenizer to use for tokenization
        seed: Random seed for shuffling
        train_ratio: Ratio of data to use for training (rest goes to validation)
        max_length: Maximum sequence length for tokenization

    Returns:
        DatasetDict with "train" and "validation" splits, tokenized and ready for training
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    train_dataset, val_dataset = dataset.train_test_split(test_size=train_ratio)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def tokenize_conversation(examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize conversations using the chat template."""
        texts = []

        for messages in examples["messages"]:
            # Apply chat template to format the conversation
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(text, str):
                texts.append(text)
            else:
                # Handle case where template returns list
                texts.append(str(text))

        # Tokenize all texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

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
