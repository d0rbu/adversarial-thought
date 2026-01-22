"""Tests for experiment modules."""

from pathlib import Path

import pytest
import torch

from exp import DATASET_DIRNAME, OUTPUT_DIRNAME
from exp.sft_finetune import ExperimentConfig, set_seed


class TestExperimentConstants:
    """Tests for experiment module constants."""

    def test_dataset_dirname_is_string(self) -> None:
        """Test that DATASET_DIRNAME is a string path."""
        assert isinstance(DATASET_DIRNAME, str)
        assert len(DATASET_DIRNAME) > 0

    def test_output_dirname_is_string(self) -> None:
        """Test that OUTPUT_DIRNAME is a string path."""
        assert isinstance(OUTPUT_DIRNAME, str)
        assert len(OUTPUT_DIRNAME) > 0

    def test_paths_are_absolute_or_relative(self) -> None:
        """Test that paths can be resolved."""
        # Should not raise
        Path(DATASET_DIRNAME).resolve()
        Path(OUTPUT_DIRNAME).resolve()


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_config(self) -> None:
        """Test that default config has sensible values."""
        cfg = ExperimentConfig()

        assert cfg.model_name == "google/gemma-3-1b-it"
        assert cfg.tokenizer_name == "google/gemma-3-1b-it"
        assert cfg.lora_enabled is True
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.batch_size == 4
        assert cfg.num_epochs == 3
        assert cfg.seed == 42

    def test_custom_config(self) -> None:
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

    def test_config_immutable_default_list(self) -> None:
        """Test that default list is not shared between instances."""
        cfg1 = ExperimentConfig()
        cfg2 = ExperimentConfig()

        cfg1.lora_target_modules.append("new_module")

        assert "new_module" not in cfg2.lora_target_modules


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self) -> None:
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

        # Very unlikely to be equal with different seeds
        assert not torch.allclose(tensor1, tensor2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_seed_cuda(self) -> None:
        """Test that set_seed works with CUDA."""
        set_seed(42)
        tensor1 = torch.randn(10, device="cuda")

        set_seed(42)
        tensor2 = torch.randn(10, device="cuda")

        assert torch.allclose(tensor1, tensor2)
