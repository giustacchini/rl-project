"""Compare histories: python src/evaluate.py <run-or-history.json> [...] --out reward.png.

This plots training episode metrics; use main.py evaluate to run held-out episodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_history(path: Path, metric: str = "total_reward", x_axis: str = "update"):
    """Read update-level metrics, retaining missing observations as plot gaps.

    Older histories are supported through mean_return when total_reward is absent.
    No completed episode means no reward measurement, rather than zero reward.
    """
    if path.is_dir():
        path = path / "history.json"
    history = json.loads(path.read_text())
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must contain a nonempty list of update records")
    updates, values = [], []
    for record in history:
        update = int(record[x_axis])
        value = record.get(metric, record.get("episode_metrics", {}).get(metric))
        if metric == "total_reward" and value is None:
            value = record.get("mean_return")
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                value = None
        if updates and update <= updates[-1]:
            raise ValueError(f"history {x_axis} values must be strictly increasing")
        updates.append(update)
        values.append(value)
    if all(value is None for value in values):
        raise ValueError(f"No measurements for {metric!r}; complete at least one training episode first")
    return updates, values


def plot_history(path: Path, output: Path, metric: str = "total_reward", show: bool = False):
    """Backwards-compatible single-history entry point."""
    return plot_histories([path], output, metric=metric, show=show)


def plot_histories(
    paths: list[Path],
    output: Path,
    metric: str = "total_reward",
    show: bool = False,
    labels: list[str] | None = None,
    x_axis: str = "update",
):
    """Overlay independent curves without truncating or interpolating their histories."""
    if not paths:
        raise ValueError("Provide at least one history")
    if labels is not None and len(labels) != len(paths):
        raise ValueError("Provide exactly one label per history")
    if x_axis not in ("update", "env_steps"):
        raise ValueError("x_axis must be update or env_steps")
    if labels is None:
        labels = [
            path.name if path.is_dir() else (path.parent.name if path.name == "history.json" else path.stem)
            for path in paths
        ]
        # Show full paths when run names alone are ambiguous.
        labels = [str(path) if labels.count(label) > 1 else label for path, label in zip(paths, labels)]
    series = []
    for path, label in zip(paths, labels):
        try:
            positions, values = read_history(path, metric, x_axis)
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        series.append((label, positions, values))

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5))
    for label, positions, values in series:
        axis.plot(positions, [float("nan") if value is None else value for value in values], marker=".", label=label)
    axis.set_xlabel("Training update" if x_axis == "update" else "Environment steps")
    axis.set_ylabel("Total episode reward" if metric == "total_reward" else metric.replace("_", " "))
    axis.set_title("Training episodes · rolling mean over last 20 completed episodes")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, nargs="+", help="one or more run directories or history JSON files")
    parser.add_argument("--labels", nargs="+", help="one legend label per history, in input order")
    parser.add_argument("--out", "--output", type=Path, help="PNG, SVG, or PDF output")
    parser.add_argument("--metric", default="total_reward", help="episode metric, e.g. snow_cleared or fraction_clean")
    parser.add_argument("--x-axis", choices=["update", "env_steps"], default="update", help="compare by updates or environment-step budget")
    parser.add_argument("--show", action="store_true", help="also open the figure interactively")
    args = parser.parse_args(argv)
    first = args.history[0]
    parent = first if first.is_dir() else first.parent
    filename = f"{args.metric.replace('/', '_')}.png"
    output = args.out or (parent / filename if len(args.history) == 1 else Path(f"comparison_{filename}"))
    try:
        plot_histories(args.history, output, args.metric, args.show, args.labels, args.x_axis)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f"saved plot to {output}")


if __name__ == "__main__":
    main()
