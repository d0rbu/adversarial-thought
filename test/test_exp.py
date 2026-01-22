"""Tests for the exp module structure.

This module contains basic tests for the experiment module's organization
and ensures the module is properly structured for imports.
"""

from __future__ import annotations


class TestExpModuleStructure:
    """Tests for exp module structure and imports."""

    def test_sft_finetune_importable(self) -> None:
        """Test that sft_finetune module is importable."""
        from exp import sft_finetune

        assert sft_finetune is not None

    def test_experiment_config_importable(self) -> None:
        """Test that ExperimentConfig is importable from sft_finetune."""
        from exp.sft_finetune import ExperimentConfig

        assert ExperimentConfig is not None

    def test_set_seed_importable(self) -> None:
        """Test that set_seed is importable from sft_finetune."""
        from exp.sft_finetune import set_seed

        assert callable(set_seed)

    def test_main_function_importable(self) -> None:
        """Test that main function is importable from sft_finetune."""
        from exp.sft_finetune import main

        assert callable(main)

    def test_load_model_and_tokenizer_importable(self) -> None:
        """Test that load_model_and_tokenizer is importable."""
        from exp.sft_finetune import load_model_and_tokenizer

        assert callable(load_model_and_tokenizer)

    def test_create_training_arguments_importable(self) -> None:
        """Test that create_training_arguments is importable."""
        from exp.sft_finetune import create_training_arguments

        assert callable(create_training_arguments)

    def test_config_to_experiment_config_importable(self) -> None:
        """Test that config_to_experiment_config is importable."""
        from exp.sft_finetune import config_to_experiment_config

        assert callable(config_to_experiment_config)
