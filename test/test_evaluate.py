"""Tests for the evaluation module.

This module tests the evaluation functionality including config handling,
metric extraction, and module structure.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from exp.evaluate import (
    EvalConfig,
    extract_metrics,
    format_metrics_by_task,
    save_results,
    save_results_yaml,
    set_seed,
)


class TestEvalConfig:
    """Tests for EvalConfig dataclass."""

    def test_default_config(self) -> None:
        """Test that default config has sensible values."""
        cfg = EvalConfig()

        assert cfg.model_name == "Qwen/Qwen3-8B"
        assert cfg.tokenizer_name is None
        assert cfg.peft_adapter_path is None
        assert "hellaswag" in cfg.tasks
        assert cfg.batch_size == "auto"
        assert cfg.device == "cuda"
        assert cfg.seed == 42

    def test_custom_config(self) -> None:
        """Test config with custom values."""
        cfg = EvalConfig(
            model_name="custom/model",
            peft_adapter_path="/path/to/adapter",
            tasks=["mmlu", "gsm8k"],
            batch_size=8,
            limit=100,
        )

        assert cfg.model_name == "custom/model"
        assert cfg.peft_adapter_path == "/path/to/adapter"
        assert cfg.tasks == ["mmlu", "gsm8k"]
        assert cfg.batch_size == 8
        assert cfg.limit == 100

    def test_tasks_default_not_shared(self) -> None:
        """Test that default tasks list is not shared between instances."""
        cfg1 = EvalConfig()
        cfg2 = EvalConfig()

        cfg1.tasks.append("new_task")

        assert "new_task" not in cfg2.tasks


class TestExtractMetrics:
    """Tests for extract_metrics function."""

    def test_extract_metrics_basic(self) -> None:
        """Test basic metric extraction from lm-eval results."""
        results = {
            "results": {
                "hellaswag": {
                    "acc,none": 0.75,
                    "acc_norm,none": 0.78,
                    "alias": "hellaswag",
                },
                "arc_easy": {
                    "acc,none": 0.80,
                    "acc_norm,none": 0.82,
                    "alias": "arc_easy",
                },
            }
        }

        metrics = extract_metrics(results)

        assert "hellaswag/acc" in metrics
        assert "hellaswag/acc_norm" in metrics
        assert "arc_easy/acc" in metrics
        assert metrics["hellaswag/acc"] == 0.75
        assert metrics["hellaswag/acc_norm"] == 0.78
        assert metrics["arc_easy/acc"] == 0.80

    def test_extract_metrics_empty_results(self) -> None:
        """Test metric extraction with empty results."""
        results: dict = {}
        metrics = extract_metrics(results)
        assert metrics == {}

    def test_extract_metrics_no_results_key(self) -> None:
        """Test metric extraction when results key is missing."""
        results = {"config": {"some": "config"}}
        metrics = extract_metrics(results)
        assert metrics == {}

    def test_extract_metrics_filters_non_numeric(self) -> None:
        """Test that non-numeric values are filtered out."""
        results = {
            "results": {
                "hellaswag": {
                    "acc,none": 0.75,
                    "alias": "hellaswag",  # string, should be filtered
                    "stderr,none": "N/A",  # string, should be filtered
                },
            }
        }

        metrics = extract_metrics(results)

        assert "hellaswag/acc" in metrics
        assert "hellaswag/alias" not in metrics
        assert "hellaswag/stderr" not in metrics

    @given(
        task_names=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
        metric_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_property_extract_metrics_returns_dict(
        self, task_names: list[str], metric_values: list[float]
    ) -> None:
        """Property: extract_metrics always returns a dictionary."""
        # Create results dict with generated task names and values
        results = {"results": {}}
        for i, task_name in enumerate(task_names):
            if i < len(metric_values):
                results["results"][task_name] = {
                    "acc,none": metric_values[i],
                }

        metrics = extract_metrics(results)

        assert isinstance(metrics, dict)
        assert all(isinstance(k, str) for k in metrics)
        assert all(isinstance(v, int | float) for v in metrics.values())

    @given(
        num_tasks=st.integers(min_value=1, max_value=5),
        metric_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_property_extract_metrics_preserves_count(
        self, num_tasks: int, metric_value: float
    ) -> None:
        """Property: Number of extracted metrics matches number of tasks."""
        results = {"results": {}}
        for i in range(num_tasks):
            results["results"][f"task_{i}"] = {"acc,none": metric_value}

        metrics = extract_metrics(results)

        # Should have one metric per task
        assert len(metrics) == num_tasks


class TestFormatMetricsByTask:
    """Tests for format_metrics_by_task function."""

    def test_organizes_by_task(self) -> None:
        """Test that metrics are organized by task."""
        metrics = {
            "hellaswag/acc": 0.75,
            "hellaswag/acc_norm": 0.78,
            "arc_easy/acc": 0.80,
        }

        by_task = format_metrics_by_task(metrics)

        assert "hellaswag" in by_task
        assert "arc_easy" in by_task
        assert by_task["hellaswag"]["acc"] == 0.75
        assert by_task["hellaswag"]["acc_norm"] == 0.78
        assert by_task["arc_easy"]["acc"] == 0.80

    def test_rounds_values(self) -> None:
        """Test that values are rounded to 4 decimal places."""
        metrics = {"task/acc": 0.123456789}

        by_task = format_metrics_by_task(metrics)

        assert by_task["task"]["acc"] == 0.1235

    def test_empty_metrics(self) -> None:
        """Test with empty metrics."""
        by_task = format_metrics_by_task({})
        assert by_task == {}

    def test_handles_no_slash(self) -> None:
        """Test metrics without task/metric format."""
        metrics = {"accuracy": 0.5}

        by_task = format_metrics_by_task(metrics)

        assert "unknown" in by_task
        assert by_task["unknown"]["accuracy"] == 0.5

    @given(
        task_names=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
        metric_names=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
        metric_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_property_format_metrics_preserves_values(
        self,
        task_names: list[str],
        metric_names: list[str],
        metric_values: list[float],
    ) -> None:
        """Property: format_metrics_by_task preserves all metric values."""
        # Create flat metrics dict
        flat_metrics: dict[str, float] = {}
        for i, task_name in enumerate(task_names):
            if i < len(metric_names) and i < len(metric_values):
                key = f"{task_name}/{metric_names[i]}"
                flat_metrics[key] = metric_values[i]

        by_task = format_metrics_by_task(flat_metrics)

        # Check that all values are preserved
        total_values = sum(len(metrics) for metrics in by_task.values())
        assert total_values == len(flat_metrics)

    @given(
        num_metrics=st.integers(min_value=1, max_value=10),
        metric_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_property_format_metrics_rounds_values(
        self, num_metrics: int, metric_value: float
    ) -> None:
        """Property: format_metrics_by_task rounds values to 4 decimal places."""
        flat_metrics = {
            f"task_{i}/metric_{i}": metric_value for i in range(num_metrics)
        }

        by_task = format_metrics_by_task(flat_metrics)

        # Check that all values are rounded to 4 decimal places
        for task_metrics in by_task.values():
            for value in task_metrics.values():
                # Value should be rounded (check by converting to string)
                value_str = f"{value:.4f}"
                assert abs(value - float(value_str)) < 1e-10


class TestSaveResultsYaml:
    """Tests for save_results_yaml function."""

    def test_creates_yaml_file(self) -> None:
        """Test that save_results_yaml creates a YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(model_name="test/model")
            metrics = {"task/acc": 0.5}

            yaml_path = save_results_yaml(metrics, tmpdir, config)

            assert yaml_path.exists()
            assert yaml_path.suffix == ".yaml"

    def test_yaml_is_readable(self) -> None:
        """Test that saved YAML is valid and readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                model_name="test/model",
                tasks=["hellaswag"],
                seed=123,
            )
            metrics = {"hellaswag/acc": 0.75, "hellaswag/acc_norm": 0.78}

            yaml_path = save_results_yaml(metrics, tmpdir, config)

            with yaml_path.open() as f:
                data = yaml.safe_load(f)

            assert "evaluation_summary" in data
            assert "results" in data
            assert data["evaluation_summary"]["model"] == "test/model"
            assert data["evaluation_summary"]["seed"] == 123
            assert data["results"]["hellaswag"]["acc"] == 0.75

    def test_yaml_includes_timestamp(self) -> None:
        """Test that YAML includes a timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(model_name="test/model")
            metrics = {}

            yaml_path = save_results_yaml(metrics, tmpdir, config)

            with yaml_path.open() as f:
                data = yaml.safe_load(f)

            assert "timestamp" in data["evaluation_summary"]


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_results_creates_both_files(self) -> None:
        """Test that save_results creates both JSON and YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(model_name="test/model")
            results = {"results": {"task": {"acc,none": 0.5}}}
            metrics = {"task/acc": 0.5}

            json_path, yaml_path = save_results(results, metrics, tmpdir, config)

            assert json_path.exists()
            assert json_path.suffix == ".json"
            assert yaml_path.exists()
            assert yaml_path.suffix == ".yaml"

    def test_save_results_json_content(self) -> None:
        """Test that saved JSON contains expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                model_name="test/model",
                tasks=["hellaswag"],
                seed=123,
            )
            results = {"results": {"hellaswag": {"acc,none": 0.75}}}
            metrics = {"hellaswag/acc": 0.75}

            json_path, _ = save_results(results, metrics, tmpdir, config)

            with json_path.open() as f:
                data = json.load(f)

            assert data["config"]["model_name"] == "test/model"
            assert data["config"]["seed"] == 123
            assert data["metrics"]["hellaswag/acc"] == 0.75

    def test_save_results_with_peft_adapter(self) -> None:
        """Test filename when using PEFT adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                model_name="test/model",
                peft_adapter_path="/path/to/my_adapter",
            )
            results = {"results": {}}
            metrics = {}

            json_path, yaml_path = save_results(results, metrics, tmpdir, config)

            assert "my_adapter" in json_path.name
            assert "my_adapter" in yaml_path.name

    def test_save_results_creates_directory(self) -> None:
        """Test that save_results creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"
            config = EvalConfig(model_name="test/model")
            results = {"results": {}}
            metrics = {}

            json_path, yaml_path = save_results(
                results, metrics, str(nested_dir), config
            )

            assert json_path.parent.exists()
            assert yaml_path.parent.exists()


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self) -> None:
        """Test that set_seed produces reproducible results."""
        import torch

        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        assert torch.allclose(tensor1, tensor2)


class TestEvaluateModuleImports:
    """Tests for evaluate module imports and structure."""

    def test_evaluate_importable(self) -> None:
        """Test that evaluate module is importable."""
        from exp import evaluate

        assert evaluate is not None

    def test_eval_config_importable(self) -> None:
        """Test that EvalConfig is importable."""
        from exp.evaluate import EvalConfig

        assert EvalConfig is not None

    def test_create_lm_eval_model_importable(self) -> None:
        """Test that create_lm_eval_model is importable."""
        from exp.evaluate import create_lm_eval_model

        assert callable(create_lm_eval_model)

    def test_run_evaluation_importable(self) -> None:
        """Test that run_evaluation is importable."""
        from exp.evaluate import run_evaluation

        assert callable(run_evaluation)

    def test_main_importable(self) -> None:
        """Test that main function is importable."""
        from exp.evaluate import main

        assert callable(main)
