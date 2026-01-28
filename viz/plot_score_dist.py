"""Visualize score distribution from a single oracle evaluation result."""

from __future__ import annotations

from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt

from viz.oracle import get_oracle_summary, get_score_distribution


def plot_score_distribution(
    yaml_path: Path,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot score distribution as a bar chart with no spaces between bars.

    Args:
        yaml_path: Path to oracle_results.yaml file
        output_path: Path to save the plot.
        title: Optional title for the plot. If None, generates from summary.
    """
    # Load data
    score_dist = get_score_distribution(yaml_path)
    summary = get_oracle_summary(yaml_path)

    # Prepare data for plotting
    scores = sorted(score_dist.keys())
    counts = [score_dist[score] for score in scores]

    # Create figure and axis
    _fig, ax = plt.subplots(figsize=(8, 6))

    # Create bar chart with no spacing (width=1.0, align='edge' with no gap)
    bars = ax.bar(
        scores,
        counts,
        width=1.0,
        align="edge",
        edgecolor="black",
        linewidth=1.0,
    )

    # Set x-axis to show all score values
    ax.set_xlabel("Judge Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # Set x-axis ticks to be centered on each bar
    ax.set_xticks([s + 0.5 for s in scores])
    ax.set_xticklabels([str(s) for s in scores])

    # Set x-axis limits to show full bars
    ax.set_xlim(min(scores) - 0.1, max(scores) + 1.1)

    # Generate title if not provided
    if title is None:
        model = summary.get("model", "Unknown")
        adapter = summary.get("target_adapter", "None")
        if adapter and adapter != "null":
            title = f"Oracle Score Distribution\n{model} ({adapter})"
        else:
            title = f"Oracle Score Distribution\n{model}"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.close()


def main(
    yaml_path: str,
    *,
    output: str | None = None,
    title: str | None = None,
) -> None:
    """Plot score distribution from oracle evaluation results.

    Parameters
    ----------
    yaml_path
        Path to oracle_results.yaml file.
    output
        Output path for the plot (default: fig/{yaml_filename_stem}.png).
    title
        Title for the plot (default: auto-generated from summary).
    """
    # Default output path based on yaml filename stem
    if output is None:
        yaml_path_obj = Path(yaml_path)
        output = f"fig/{yaml_path_obj.stem}.png"

    plot_score_distribution(
        yaml_path=Path(yaml_path),
        output_path=Path(output),
        title=title,
    )


if __name__ == "__main__":
    cyclopts.run(main)
