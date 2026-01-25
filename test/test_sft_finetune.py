"""Tests for exp/sft_finetune.py - SFT finetuning experiment.

This module contains both unit tests and property-based tests for the
ExperimentConfig, set_seed, and training configuration utilities.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from core.dtype import get_dtype
from exp.sft_finetune import ExperimentConfig, set_seed

# =============================================================================
# Strategies for property-based testing
# =============================================================================

# Strategy for seeds
seeds = st.integers(min_value=0, max_value=2**31 - 1)

# Strategy for positive integers (for config fields)
positive_ints = st.integers(min_value=1, max_value=1000)

# Strategy for floats between 0 and 1
unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Strategy for small positive floats (learning rates, etc.)
small_positive_floats = st.floats(
    min_value=1e-10, max_value=1.0, allow_nan=False, allow_infinity=False
)


# =============================================================================
# Unit tests for ExperimentConfig
# =============================================================================


class TestExperimentConfig:
    """Unit tests for ExperimentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default config has sensible values."""
        cfg = ExperimentConfig()

        assert cfg.model_name == "Qwen/Qwen3-8B"
        assert cfg.tokenizer_name == "Qwen/Qwen3-8B"
        assert cfg.lora_enabled is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.batch_size == 4
        assert cfg.num_epochs == 3
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        """Test config with custom values."""
        cfg = ExperimentConfig(
            model_name="custom/model",
            lora_enabled=False,
            batch_size=8,
            seed=123,
        )

        assert cfg.model_name == "custom/model"
        assert cfg.lora_enabled is False
        assert cfg.batch_size == 8
        assert cfg.seed == 123

    def test_lora_target_modules_default(self) -> None:
        """Test that LoRA target modules have sensible defaults."""
        cfg = ExperimentConfig()

        assert "q_proj" in cfg.lora_target_modules
        assert "k_proj" in cfg.lora_target_modules
        assert "v_proj" in cfg.lora_target_modules
        assert "o_proj" in cfg.lora_target_modules

    def test_default_list_not_shared(self) -> None:
        """Test that default list is not shared between instances."""
        cfg1 = ExperimentConfig()
        cfg2 = ExperimentConfig()

        cfg1.lora_target_modules.append("new_module")

        assert "new_module" not in cfg2.lora_target_modules

    def test_dtype_default(self) -> None:
        """Test default dtype is bfloat16."""
        cfg = ExperimentConfig()
        assert cfg.dtype == "bfloat16"

    def test_training_hyperparameters_defaults(self) -> None:
        """Test training hyperparameters have sensible defaults."""
        cfg = ExperimentConfig()

        assert cfg.learning_rate == 2e-5
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_ratio == 0.03
        assert cfg.max_grad_norm == 1.0
        assert cfg.gradient_checkpointing is True


# =============================================================================
# Property-based tests for ExperimentConfig
# =============================================================================


class TestExperimentConfigProperties:
    """Property-based tests for ExperimentConfig dataclass."""

    @given(
        lora_r=positive_ints,
        lora_alpha=positive_ints,
        batch_size=positive_ints,
        num_epochs=positive_ints,
        seed=seeds,
    )
    def test_stores_values_correctly(
        self,
        lora_r: int,
        lora_alpha: int,
        batch_size: int,
        num_epochs: int,
        seed: int,
    ) -> None:
        """Config should store all provided values exactly."""
        cfg = ExperimentConfig(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            batch_size=batch_size,
            num_epochs=num_epochs,
            seed=seed,
        )
        assert cfg.lora_r == lora_r
        assert cfg.lora_alpha == lora_alpha
        assert cfg.batch_size == batch_size
        assert cfg.num_epochs == num_epochs
        assert cfg.seed == seed

    @given(st.integers(min_value=0, max_value=100))
    def test_default_list_isolation(self, n_configs: int) -> None:
        """Each config instance should have its own mutable list."""
        configs = [ExperimentConfig() for _ in range(n_configs)]

        if configs:
            configs[0].lora_target_modules.append("test_module")

            for cfg in configs[1:]:
                assert "test_module" not in cfg.lora_target_modules

    @given(
        learning_rate=small_positive_floats,
        weight_decay=unit_floats,
        warmup_ratio=unit_floats,
    )
    def test_accepts_valid_float_hyperparams(
        self,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
    ) -> None:
        """Config should accept any valid float hyperparameters."""
        cfg = ExperimentConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
        )
        assert cfg.learning_rate == learning_rate
        assert cfg.weight_decay == weight_decay
        assert cfg.warmup_ratio == warmup_ratio

    @given(st.sampled_from(["bfloat16", "float16", "float32"]))
    def test_dtype_valid_for_get_dtype(self, dtype: str) -> None:
        """Config dtype should be valid for get_dtype conversion."""
        cfg = ExperimentConfig(dtype=dtype)
        result = get_dtype(cfg.dtype)
        assert isinstance(result, torch.dtype)

    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        grad_accum=st.integers(min_value=1, max_value=32),
    )
    def test_effective_batch_size(self, batch_size: int, grad_accum: int) -> None:
        """Effective batch size should be batch_size * gradient_accumulation_steps."""
        cfg = ExperimentConfig(
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
        )
        effective = cfg.batch_size * cfg.gradient_accumulation_steps
        assert effective == batch_size * grad_accum

    @given(st.sampled_from(["cuda", "cpu"]))
    def test_device_setting(self, device: str) -> None:
        """Device setting should be stored correctly."""
        cfg = ExperimentConfig(device=device)
        assert cfg.device == device

    @given(train_ratio=st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    def test_train_ratio_valid(self, train_ratio: float) -> None:
        """Train ratio should be stored and be in valid range."""
        cfg = ExperimentConfig(train_ratio=train_ratio)
        assert 0 < cfg.train_ratio < 1


# =============================================================================
# Unit tests for set_seed
# =============================================================================


class TestSetSeed:
    """Unit tests for set_seed function."""

    def test_reproducibility(self) -> None:
        """Test that set_seed produces reproducible results."""
        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        assert torch.allclose(tensor1, tensor2)

    def test_different_seeds_different_results(self) -> None:
        """Test that different seeds produce different results."""
        set_seed(42)
        tensor1 = torch.randn(100)

        set_seed(123)
        tensor2 = torch.randn(100)

        assert not torch.allclose(tensor1, tensor2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_reproducibility(self) -> None:
        """Test that set_seed works with CUDA."""
        set_seed(42)
        tensor1 = torch.randn(10, device="cuda")

        set_seed(42)
        tensor2 = torch.randn(10, device="cuda")

        assert torch.allclose(tensor1, tensor2)


# =============================================================================
# Property-based tests for set_seed
# =============================================================================


class TestSetSeedProperties:
    """Property-based tests for set_seed function."""

    @given(seeds)
    @settings(max_examples=50)
    def test_torch_reproducibility(self, seed: int) -> None:
        """Setting the same seed twice produces identical torch random numbers."""
        set_seed(seed)
        tensor1 = torch.randn(100)

        set_seed(seed)
        tensor2 = torch.randn(100)

        assert torch.allclose(tensor1, tensor2)

    @given(seeds)
    @settings(max_examples=50)
    def test_numpy_reproducibility(self, seed: int) -> None:
        """Setting the same seed twice produces identical numpy random numbers."""
        set_seed(seed)
        arr1 = np.random.rand(100)

        set_seed(seed)
        arr2 = np.random.rand(100)

        assert np.allclose(arr1, arr2)

    @given(seeds)
    @settings(max_examples=50)
    def test_python_random_reproducibility(self, seed: int) -> None:
        """Setting the same seed twice produces identical Python random numbers."""
        set_seed(seed)
        nums1 = [random.random() for _ in range(100)]

        set_seed(seed)
        nums2 = [random.random() for _ in range(100)]

        assert nums1 == nums2

    @given(st.tuples(seeds, seeds).filter(lambda x: x[0] != x[1]))
    @settings(max_examples=30)
    def test_different_seeds_differ(self, seed_pair: tuple[int, int]) -> None:
        """Different seeds should produce different random sequences."""
        seed1, seed2 = seed_pair

        set_seed(seed1)
        tensor1 = torch.randn(1000)

        set_seed(seed2)
        tensor2 = torch.randn(1000)

        assert not torch.allclose(tensor1, tensor2)
