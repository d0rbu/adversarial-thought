"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_random_state() -> None:
    """Reset random state before each test for reproducibility."""
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]


@pytest.fixture
def device() -> str:
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
