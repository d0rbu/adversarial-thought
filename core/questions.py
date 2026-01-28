"""Questions for probing activation oracles.

Simple train/val question lists. Questions are designed to be:
1. Answerable from the context (clear ground truth exists)
2. Evaluatable by an LLM judge (1-5 accuracy score)
"""

from __future__ import annotations

# =============================================================================
# Training questions (for adversarial training)
# These questions have clear, verifiable answers from the context.
# =============================================================================

TRAIN_QUESTIONS: list[str] = [
    # Token/text prediction (verifiable from context)
    "What is the last word the user or assistant typed?",
    # Content extraction (verifiable from context)
    "What is the main topic of the last sentence?",
    "What named entities (people, places, organizations) appear in the text?",
    "What question did the user ask, if any?",
    # Factual grounding (verifiable from context)
    "What facts or claims are stated in the text?",
    "What action is the user requesting?",
    "What is the subject of the conversation?",
    # Format/structure (verifiable from context)
    "Is the text a question, statement, or command?",
    "What language is this text written in?",
    "Is there code in the text? If so, what programming language?",
]

# =============================================================================
# Validation questions (held out for evaluation)
# Similar style to training, but distinct questions.
# =============================================================================

VAL_QUESTIONS: list[str] = [
    # Token/text prediction
    "Can you predict the last three words?",
    "What is the final character or symbol?",
    "Is the text complete or cut off mid-sentence?",
    # Content extraction
    "What dates or times are referenced?",
    "What is being described or explained?",
    # Factual grounding
    "What is the user trying to accomplish?",
    "What domain or field does this text relate to?",
    "What key terms or jargon appear?",
    # Format/structure
    "How many paragraphs or sections are there?",
    "Is this formal or informal writing?",
    "Does this text contain a list or enumeration?",
]


def get_train_questions() -> list[str]:
    """Get training questions."""
    return TRAIN_QUESTIONS.copy()


def get_val_questions() -> list[str]:
    """Get validation questions."""
    return VAL_QUESTIONS.copy()


def to_chat_message(question: str) -> list[dict[str, str]]:
    """Convert a question string to chat message format.

    Args:
        question: The question to ask the activation oracle.

    Returns:
        A list with a single user message dict.

    Example:
        >>> to_chat_message("What is the topic?")
        [{"role": "user", "content": "What is the topic?"}]
    """
    return [{"role": "user", "content": question}]
