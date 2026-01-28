"""Visualizations for lm-eval-harness results (avg bar, grouped bars, radar chart)."""

from __future__ import annotations

from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt
import numpy as np

from viz.lmeval import average_score, extract_task_scores, get_lmeval_summary


def _default_labels(yaml_paths: list[Path]) -> list[str]:
    labels: list[str] = []
    for p in yaml_paths:
        s = get_lmeval_summary(p)
        if s.model:
            labels.append(s.model.split("/")[-1])
        else:
            labels.append(p.parent.name if p.parent.name not in {"", "."} else p.stem)
    return labels


def plot_lmeval_average_bar(
    *,
    yaml_paths: list[Path],
    labels: list[str],
    output_path: Path,
    tasks: tuple[str, ...] | None,
    title: str | None,
) -> None:
    scores = []
    for p in yaml_paths:
        task_scores = extract_task_scores(p, tasks=tasks)
        scores.append(average_score(task_scores))

    _fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 6))
    x = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax.bar(
        x,
        scores,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Average lm-eval score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title(title or "Average lm-eval score (across tasks)", fontweight="bold")

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_lmeval_grouped_bars(
    *,
    yaml_paths: list[Path],
    labels: list[str],
    output_path: Path,
    tasks: tuple[str, ...] | None,
    title: str | None,
) -> None:
    per_model: list[dict[str, float]] = [
        extract_task_scores(p, tasks=tasks) for p in yaml_paths
    ]

    # Union of tasks actually present across models (keeps plot robust).
    task_names = sorted({k for d in per_model for k in d})
    if not task_names:
        raise ValueError("No lm-eval task scores found to plot")

    n_groups = len(task_names)
    n_bars = len(labels)
    bar_width = 0.8 / max(1, n_bars)

    _fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_groups), 6))
    x = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    for i, (label, d, color) in enumerate(zip(labels, per_model, colors)):
        vals = [d.get(t, np.nan) for t in task_names]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            np.nan_to_num(vals, nan=0.0),
            bar_width,
            label=label,
            edgecolor="black",
            linewidth=0.5,
            color=color,
        )

        # Label only if present (not NaN)
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{float(v):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title(title or "lm-eval scores by task", fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_lmeval_radar(
    *,
    yaml_paths: list[Path],
    labels: list[str],
    output_path: Path,
    tasks: tuple[str, ...] | None,
    title: str | None,
) -> None:
    per_model: list[dict[str, float]] = [
        extract_task_scores(p, tasks=tasks) for p in yaml_paths
    ]
    task_names = sorted({k for d in per_model for k in d})
    if not task_names:
        raise ValueError("No lm-eval task scores found to plot")

    # Radar needs closed loop.
    angles = np.linspace(0, 2 * np.pi, len(task_names), endpoint=False).tolist()
    angles += angles[:1]

    _fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for label, d, color in zip(labels, per_model, colors):
        vals = [d.get(t, 0.0) for t in task_names]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(title or "lm-eval radar chart", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    *yaml_paths: str,
    labels: str | None = None,
    output_dir: str = "fig",
    output_prefix: str | None = None,
    tasks: str | None = None,
    title_avg: str | None = None,
    title_grouped: str | None = None,
    title_radar: str | None = None,
) -> None:
    """Generate three lm-eval figures from one or more lm-eval YAML files.

    Parameters
    ----------
    *yaml_paths
        One or more paths to `eval_*.yaml` produced by `exp.evaluate`.
    labels
        Optional comma-separated labels (one per YAML). Default: derived from metadata.
    output_dir
        Directory to write figures into (default: fig).
    output_prefix
        Optional filename prefix (without extension). If omitted, uses the first YAML stem.
        Example: "baseline_vs_sft" -> fig/lmeval_avg_baseline_vs_sft.png, etc.
    tasks
        Optional comma-separated task list (e.g. "hellaswag,winogrande,mmlu,xquad").
        If omitted, uses each YAML's `evaluation_summary.tasks`.
    title_avg/title_grouped/title_radar
        Optional titles for each plot.
    """
    if not yaml_paths:
        raise ValueError("At least one YAML path must be provided")

    yaml_path_objs = [Path(p) for p in yaml_paths]

    parsed_labels: list[str]
    if labels is None:
        parsed_labels = _default_labels(yaml_path_objs)
    else:
        parsed_labels = [s.strip() for s in labels.split(",")]
        if len(parsed_labels) != len(yaml_path_objs):
            raise ValueError(
                f"Number of labels ({len(parsed_labels)}) must match "
                f"number of YAML paths ({len(yaml_path_objs)})"
            )

    parsed_tasks: tuple[str, ...] | None = None
    if tasks is not None:
        parsed_tasks = tuple(t.strip() for t in tasks.split(",") if t.strip())

    out_dir = Path(output_dir)
    stem = output_prefix or yaml_path_objs[0].stem

    plot_lmeval_average_bar(
        yaml_paths=yaml_path_objs,
        labels=parsed_labels,
        output_path=out_dir / f"lmeval_avg_{stem}.png",
        tasks=parsed_tasks,
        title=title_avg,
    )
    plot_lmeval_grouped_bars(
        yaml_paths=yaml_path_objs,
        labels=parsed_labels,
        output_path=out_dir / f"lmeval_grouped_{stem}.png",
        tasks=parsed_tasks,
        title=title_grouped,
    )
    plot_lmeval_radar(
        yaml_paths=yaml_path_objs,
        labels=parsed_labels,
        output_path=out_dir / f"lmeval_radar_{stem}.png",
        tasks=parsed_tasks,
        title=title_radar,
    )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    cyclopts.run(main)
