"""Visualize score distribution from multiple oracle evaluation results in a grouped bar chart."""

from __future__ import annotations

from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt
import numpy as np

from viz.oracle import get_oracle_summary, get_score_distribution


def plot_score_distribution_grouped(
    yaml_paths: list[Path],
    output_path: Path,
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Plot score distributions from multiple oracle results as grouped bar charts.

    Args:
        yaml_paths: List of paths to oracle_results.yaml files
        output_path: Path to save the plot.
        labels: Optional list of labels for each YAML file. If None, uses directory names.
        title: Optional title for the plot. If None, generates from summaries.
    """
    if not yaml_paths:
        raise ValueError("At least one YAML path must be provided")

    # Load all score distributions
    score_dists = []
    summaries = []
    for yaml_path in yaml_paths:
        score_dists.append(get_score_distribution(yaml_path))
        summaries.append(get_oracle_summary(yaml_path))

    # Generate labels if not provided
    if labels is None:
        labels = []
        for yaml_path in yaml_paths:
            # Use parent directory name as label
            label = yaml_path.parent.name
            if label == "." or not label:
                label = yaml_path.stem
            labels.append(label)

    if len(labels) != len(yaml_paths):
        raise ValueError("Number of labels must match number of YAML paths")

    # Get all unique scores across all distributions
    all_scores = set()
    for dist in score_dists:
        all_scores.update(dist.keys())
    scores = sorted(all_scores)

    # Prepare data for grouped bars
    # Each group represents a score value, each bar in the group represents a YAML file
    n_groups = len(scores)
    n_bars = len(yaml_paths)
    bar_width = 0.8 / n_bars  # Total width of 0.8 divided by number of bars

    # Create figure and axis
    _, ax = plt.subplots(figsize=(max(8, 2 * n_groups), 6))

    # Create grouped bars
    x = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    for i, (label, dist, color) in enumerate(zip(labels, score_dists, colors)):
        counts = [dist.get(score, 0) for score in scores]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            counts,
            bar_width,
            label=label,
            edgecolor="black",
            linewidth=0.5,
            color=color,
        )

        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            if count > 0:  # Only label non-zero bars
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Set x-axis labels
    ax.set_xlabel("Judge Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scores])

    # Set x-axis limits
    ax.set_xlim(-0.5, n_groups - 0.5)

    # Generate title if not provided
    if title is None:
        if len(summaries) == 1:
            model = summaries[0].get("model", "Unknown")
            title = f"Oracle Score Distribution\n{model}"
        else:
            title = "Oracle Score Distribution Comparison"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend
    ax.legend(loc="best", fontsize=10)

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.close()


def main(
    *yaml_paths: str,
    labels: str | None = None,
    output: str | None = None,
    title: str | None = None,
) -> None:
    """Plot score distributions from multiple oracle evaluation results.

    Parameters
    ----------
    *yaml_paths
        One or more paths to oracle_results.yaml files.
    labels
        Optional comma-separated labels for each YAML file.
        If None, uses directory names.
    output
        Output path for the plot (default: fig/grouped_{first_yaml_stem}.png).
    title
        Title for the plot (default: auto-generated from summaries).
    """
    if not yaml_paths:
        raise ValueError("At least one YAML path must be provided")

    # Convert string paths to Path objects
    yaml_path_objs = [Path(path) for path in yaml_paths]

    # Default output path based on first yaml filename stem
    if output is None:
        output = f"fig/grouped_{yaml_path_objs[0].stem}.png"

    # Parse labels if provided as comma-separated string
    parsed_labels: list[str] | None = None
    if labels is not None:
        parsed_labels = [label.strip() for label in labels.split(",")]
        if len(parsed_labels) != len(yaml_path_objs):
            raise ValueError(
                f"Number of labels ({len(parsed_labels)}) must match "
                f"number of YAML paths ({len(yaml_path_objs)})"
            )

    plot_score_distribution_grouped(
        yaml_paths=yaml_path_objs,
        output_path=Path(output),
        labels=parsed_labels,
        title=title,
    )


if __name__ == "__main__":
    cyclopts.run(main)
