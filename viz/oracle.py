"""Helper functions for loading and processing oracle evaluation results."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_oracle_yaml(yaml_path: Path) -> dict:
    """Load oracle results from YAML file.

    Args:
        yaml_path: Path to oracle_results.yaml file

    Returns:
        Dictionary containing the YAML data

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    return data


def get_score_distribution(yaml_path: Path) -> dict[int, int]:
    """Extract score distribution from oracle results YAML.

    Args:
        yaml_path: Path to oracle_results.yaml file

    Returns:
        Dictionary mapping score (int) to count (int)

    Raises:
        KeyError: If score_distribution is missing from the YAML
        ValueError: If score keys cannot be converted to integers
    """
    data = load_oracle_yaml(yaml_path)

    if "metrics" not in data:
        raise KeyError("Missing 'metrics' key in YAML file")
    if "score_distribution" not in data["metrics"]:
        raise KeyError("Missing 'score_distribution' in metrics")

    dist = data["metrics"]["score_distribution"]

    # Convert string keys to integers and ensure values are integers
    result = {}
    for score_str, count in dist.items():
        try:
            score = int(score_str)
            result[score] = int(count)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid score or count in distribution: {score_str}={count}"
            ) from e

    return result


def get_oracle_summary(yaml_path: Path) -> dict:
    """Extract summary information from oracle results YAML.

    Args:
        yaml_path: Path to oracle_results.yaml file

    Returns:
        Dictionary containing oracle evaluation summary
    """
    data = load_oracle_yaml(yaml_path)
    return data.get("oracle_evaluation_summary", {})


def get_metrics(yaml_path: Path) -> dict:
    """Extract metrics from oracle results YAML.

    Args:
        yaml_path: Path to oracle_results.yaml file

    Returns:
        Dictionary containing metrics
    """
    data = load_oracle_yaml(yaml_path)
    return data.get("metrics", {})
