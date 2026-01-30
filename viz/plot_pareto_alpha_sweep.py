"""Plot Pareto frontier for adversarial alpha sweeps.

X-axis: oracle validation performance (mean judge score, optionally normalized).
Y-axis: lm-eval-harness performance (average across tasks in the lm-eval YAML).

The alpha sweep points are connected in increasing-alpha order and each point is
annotated with its alpha value. Baseline points for the base model and the SFT
model can also be included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cyclopts
import matplotlib.pyplot as plt

from viz.lmeval import average_score, extract_task_scores
from viz.oracle import get_metrics, get_oracle_summary


@dataclass(frozen=True)
class Point:
    label: str
    x: float
    y: float
    kind: str  # "alpha" | "base" | "sft"


def _alpha_to_dir_suffix(alpha: float) -> str:
    # Matches `script/run_adversarial_sft_alpha_sweep.sh` naming.
    # Example: 0.004 -> "0_004"
    return str(alpha).replace(".", "_")


def _find_unique(paths: list[Path], *, what: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Could not find {what}")
    if len(paths) > 1:
        shown = "\n".join(f"- {p}" for p in paths[:10])
        more = "" if len(paths) <= 10 else f"\n... and {len(paths) - 10} more"
        raise ValueError(f"Found multiple candidates for {what}:\n{shown}{more}")
    return paths[0]


def _find_lmeval_yaml_for_adapter(out_root: Path, adapter_path: str | None) -> Path:
    """Find eval YAML produced by `exp.evaluate` for a given adapter (or base)."""
    if adapter_path:
        adapter_name = Path(adapter_path).name
        candidates = sorted(out_root.glob(f"**/eval_{adapter_name}.yaml"))
        return _find_unique(
            candidates,
            what=f"lm-eval YAML for adapter {adapter_path!r} (pattern eval_{adapter_name}.yaml under {out_root})",
        )

    # Base model eval: filename is eval_{model_name_with_slashes_replaced}.yaml
    # We don't know model_name here, so look for eval_*.yaml with adapter=None.
    candidates = sorted(out_root.glob("**/eval_*.yaml"))
    base_candidates: list[Path] = []
    for p in candidates:
        try:
            # Reuse loader logic from viz.lmeval by reading via helper.
            # We avoid importing load_lmeval_yaml to keep the interface small.
            scores = extract_task_scores(p)
            if scores:  # file looks like an lm-eval artifact
                pass
        except Exception:
            continue

        # Heuristic: base eval YAML has `evaluation_summary.adapter: null`
        # We'll read the YAML via extract_task_scores() already succeeded; now parse summary.
        from viz.lmeval import get_lmeval_summary

        s = get_lmeval_summary(p)
        if s.adapter is None:
            base_candidates.append(p)

    return _find_unique(
        base_candidates,
        what=f"base-model lm-eval YAML (evaluation_summary.adapter null) under {out_root}",
    )


def _find_oracle_yaml_for_target_adapter(
    out_root: Path, target_adapter_path: str | None
) -> Path:
    """Find oracle_results.yaml produced by `exp.run_oracle` for a target adapter."""
    candidates = sorted(out_root.glob("**/oracle_results.yaml"))
    matched: list[Path] = []
    for p in candidates:
        try:
            summary = get_oracle_summary(p)
        except Exception:
            continue
        if summary.get("target_adapter") == target_adapter_path:
            matched.append(p)
    return _find_unique(
        matched,
        what=f"oracle_results.yaml with target_adapter={target_adapter_path!r} under {out_root}",
    )


def _oracle_score(yaml_path: Path, *, normalize: bool) -> float:
    metrics = get_metrics(yaml_path)
    mean_score = metrics.get("mean_score")
    if not isinstance(mean_score, int | float):
        raise ValueError(f"Missing/invalid metrics.mean_score in {yaml_path}")
    mean_score_f = float(mean_score)
    if normalize:
        return mean_score_f / 5.0
    return mean_score_f


def _lmeval_avg(yaml_path: Path) -> float:
    return average_score(extract_task_scores(yaml_path))


def plot_pareto(
    *,
    out_root: Path,
    alphas: list[float],
    alpha_adapter_prefix: str = "out/adversarial_sft_alpha_",
    include_base: bool = True,
    include_sft: bool = True,
    sft_adapter_path: str = "out/sft_baseline",
    oracle_normalize: bool = True,
    output_path: Path,
    title: str | None = None,
) -> None:
    out_root = Path(out_root)

    points: list[Point] = []

    # Base point(s)
    if include_base:
        oracle_yaml = _find_oracle_yaml_for_target_adapter(out_root, None)
        lmeval_yaml = _find_lmeval_yaml_for_adapter(out_root, None)
        points.append(
            Point(
                label="base",
                x=_oracle_score(oracle_yaml, normalize=oracle_normalize),
                y=_lmeval_avg(lmeval_yaml),
                kind="base",
            )
        )

    if include_sft:
        oracle_yaml = _find_oracle_yaml_for_target_adapter(out_root, sft_adapter_path)
        lmeval_yaml = _find_lmeval_yaml_for_adapter(out_root, sft_adapter_path)
        points.append(
            Point(
                label="sft",
                x=_oracle_score(oracle_yaml, normalize=oracle_normalize),
                y=_lmeval_avg(lmeval_yaml),
                kind="sft",
            )
        )

    # Alpha sweep points (ordered by alpha).
    # Some runs may have oracle results but no lm-eval; we record those as (alpha, x) for dashed vertical lines.
    alpha_points: list[Point] = []
    oracle_only: list[tuple[float, float]] = []  # (alpha, oracle_x)
    for a in sorted(alphas):
        adapter_path = f"{alpha_adapter_prefix}{_alpha_to_dir_suffix(a)}"
        try:
            oracle_yaml = _find_oracle_yaml_for_target_adapter(out_root, adapter_path)
        except (FileNotFoundError, ValueError):
            continue
        x = _oracle_score(oracle_yaml, normalize=oracle_normalize)
        try:
            lmeval_yaml = _find_lmeval_yaml_for_adapter(out_root, adapter_path)
            y = _lmeval_avg(lmeval_yaml)
        except (FileNotFoundError, ValueError):
            oracle_only.append((a, x))
            continue
        alpha_points.append(
            Point(
                label=f"{a:g}",
                x=x,
                y=y,
                kind="alpha",
            )
        )

    points.extend(alpha_points)

    _fig, ax = plt.subplots(figsize=(8, 6))

    # Plot alpha sweep as connected line.
    xs = [p.x for p in alpha_points]
    ys = [p.y for p in alpha_points]
    ax.plot(xs, ys, "-o", linewidth=2, markersize=6, label="alpha sweep")

    # Label each alpha point.
    for p in alpha_points:
        ax.annotate(
            p.label,
            (p.x, p.y),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
        )

    # Oracle-only runs (no lm-eval): dashed vertical line at oracle score.
    for i, (a, x) in enumerate(oracle_only):
        ax.axvline(
            x=x,
            color="gray",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label="oracle only (no lm-eval)" if i == 0 else None,
        )
        y_top = ax.get_ylim()[1]
        ax.annotate(
            f"{a:g} (no lm-eval)",
            (x, y_top),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="gray",
            rotation=90,
            rotation_mode="anchor",
        )

    # Extend x-axis so oracle-only vertical lines are visible.
    if oracle_only:
        x_min, x_max = ax.get_xlim()
        ox_vals = [x for _, x in oracle_only]
        ax.set_xlim(min(x_min, *ox_vals), max(x_max, *ox_vals))

    # Baselines (not connected)
    for p, color, marker in [
        (next((q for q in points if q.kind == "base"), None), "black", "s"),
        (next((q for q in points if q.kind == "sft"), None), "tab:green", "D"),
    ]:
        if p is None:
            continue
        ax.scatter([p.x], [p.y], color=color, marker=marker, s=70, label=p.label)
        ax.annotate(
            p.label,
            (p.x, p.y),
            textcoords="offset points",
            xytext=(6, -10),
            ha="left",
            fontsize=9,
            color=color,
        )

    ax.set_xlabel(
        "Oracle validation performance (mean score / 5)"
        if oracle_normalize
        else "Oracle validation performance (mean score)"
    )
    ax.set_ylabel("lm-eval performance (avg across tasks)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(
        title or "Pareto frontier: lm-eval vs oracle (alpha sweep)", fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(
    *,
    out_root: str = "out",
    alphas: str = "0.001,0.004,0.01,0.04,0.1",
    alpha_adapter_prefix: str = "out/adversarial_sft_alpha_",
    include_base: bool = True,
    include_sft: bool = True,
    sft_adapter_path: str = "out/sft_baseline",
    oracle_normalize: bool = True,
    output: str = "fig/pareto_alpha_sweep.png",
    title: str | None = None,
) -> None:
    """Plot Pareto frontier for an adversarial alpha sweep.

    Parameters
    ----------
    out_root
        Root directory to search for artifacts (default: out).
        The script searches recursively for:
        - `eval_*.yaml` (from `exp.evaluate`)
        - `oracle_results.yaml` (from `exp.run_oracle`)
    alphas
        Comma-separated alpha values in sweep order (the script sorts numerically).
    alpha_adapter_prefix
        Adapter path prefix used during training (default matches the sweep script):
        `out/adversarial_sft_alpha_` + alpha with '.' replaced by '_'.
    include_base/include_sft
        Whether to include base and SFT baseline points.
    sft_adapter_path
        Adapter path for the SFT baseline (default: out/sft_baseline).
    oracle_normalize
        If true, divide oracle mean score by 5 so x-axis is in [0,1].
    output
        Output plot path.
    title
        Optional plot title.
    """
    alpha_list = []
    for s in alphas.split(","):
        s = s.strip()
        if not s:
            continue
        alpha_list.append(float(s))
    if not alpha_list:
        raise ValueError("No alphas provided")

    plot_pareto(
        out_root=Path(out_root),
        alphas=alpha_list,
        alpha_adapter_prefix=alpha_adapter_prefix,
        include_base=include_base,
        include_sft=include_sft,
        sft_adapter_path=sft_adapter_path,
        oracle_normalize=oracle_normalize,
        output_path=Path(output),
        title=title,
    )
    print(f"Wrote figure to: {output}")


if __name__ == "__main__":
    cyclopts.run(main)
