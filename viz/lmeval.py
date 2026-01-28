"""Helper functions for loading and processing lm-eval-harness evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LmEvalSummary:
    """Small summary extracted from an lm-eval YAML artifact."""

    model: str | None
    adapter: str | None
    tasks: tuple[str, ...]
    timestamp: str | None


def load_lmeval_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load an lm-eval YAML file produced by `exp.evaluate`.

    Expected shape (example):
      evaluation_summary: { model, adapter, tasks, ... }
      results: { task_or_subtask: {metric: value, ...}, ... }
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root type: {type(data)!r}")
    return data  # type: ignore[return-value]


def get_lmeval_summary(yaml_path: Path) -> LmEvalSummary:
    """Extract basic metadata from an lm-eval YAML."""
    data = load_lmeval_yaml(yaml_path)
    summary = data.get("evaluation_summary", {}) or {}
    tasks = summary.get("tasks", []) or []
    if not isinstance(tasks, list):
        tasks = []
    return LmEvalSummary(
        model=summary.get("model"),
        adapter=summary.get("adapter"),
        tasks=tuple(str(t) for t in tasks),
        timestamp=summary.get("timestamp"),
    )


def _preferred_metric_value(
    metrics: dict[str, Any],
    *,
    metric_preference: tuple[str, ...],
) -> float | None:
    for key in metric_preference:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, int | float):
                return float(value)
    return None


def extract_task_scores(
    yaml_path: Path,
    *,
    tasks: tuple[str, ...] | None = None,
    metric_preference: tuple[str, ...] = ("acc_norm", "acc", "exact_match", "f1"),
) -> dict[str, float]:
    """Extract a single scalar score per "task" for plotting.

    Rules:
    - If `tasks` is None, uses `evaluation_summary.tasks` if present; otherwise derives
      from `results` keys (top-level prefixes).
    - For each task:
      - If `results[task]` exists, use that task entry.
      - Else if there are subtasks (keys starting with f"{task}_"), average their
        preferred metric (e.g. average xquad language F1s).
    - Metric chosen per entry is the first available in `metric_preference`.
    """
    data = load_lmeval_yaml(yaml_path)
    results = data.get("results", {}) or {}
    if not isinstance(results, dict):
        raise ValueError("Missing/invalid 'results' in lm-eval YAML")

    if tasks is None:
        summary_tasks = get_lmeval_summary(yaml_path).tasks
        if summary_tasks:
            tasks = summary_tasks
        else:
            # Heuristic: take unique prefixes before first underscore.
            prefixes: set[str] = set()
            for k in results:
                if not isinstance(k, str):
                    continue
                prefixes.add(k.split("_", 1)[0])
            tasks = tuple(sorted(prefixes))

    out: dict[str, float] = {}
    for task in tasks:
        if task in results and isinstance(results[task], dict):
            v = _preferred_metric_value(
                results[task], metric_preference=metric_preference
            )
            if v is not None:
                out[task] = v
            continue

        # Try subtasks, e.g. xquad_en/xquad_es, hendrycks_math_*, mmlu_*
        sub_vals: list[float] = []
        prefix = f"{task}_"
        for name, metrics in results.items():
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            if not isinstance(metrics, dict):
                continue
            v = _preferred_metric_value(metrics, metric_preference=metric_preference)
            if v is not None:
                sub_vals.append(v)
        if sub_vals:
            out[task] = sum(sub_vals) / len(sub_vals)

    return out


def average_score(task_scores: dict[str, float]) -> float:
    """Average across tasks."""
    if not task_scores:
        raise ValueError("No task scores to average")
    return sum(task_scores.values()) / len(task_scores)
