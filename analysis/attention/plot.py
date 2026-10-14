"""Replot the paper's cached Copy/Sort/Maze/ImageNet attention as four task rows."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis/attention_layout"
DESTINATIONS = ("Sink", "LTM", "STM")


def collect_curves(algorithm_path, visual_dir):
    with algorithm_path.open() as handle:
        algorithm = list(csv.DictReader(handle))
    sources = [algorithm_path]
    tasks = []
    for task in ("Copy", "Sort"):
        rows = sorted(
            (row for row in algorithm if row["task"] == task),
            key=lambda row: int(row["timestep"]),
        )
        x = np.array([int(row["timestep"]) for row in rows])
        budgets = {
            query: np.array(
                [
                    [
                        float(row[f"{query.lower()}_to_{destination.lower()}"])
                        for row in rows
                    ]
                    for destination in DESTINATIONS
                ]
            )
            for query in ("LTM", "STM")
        }
        boundary = (
            min(int(row["timestep"]) for row in rows if row["phase"] == "output") - 0.5
        )
        tasks.append((task, x, budgets, boundary))

    for task, title in (("Maze", "Maze"), ("ImageNet100", "ImageNet-100")):
        grouped = defaultdict(list)
        for seed in range(3):
            source = visual_dir / f"{task}_seed{seed}.csv"
            sources.append(source)
            with source.open() as handle:
                for row in csv.DictReader(handle):
                    if row["condition"] == "intact":
                        assert int(row["num_examples"]) == 5000
                        grouped[
                            (int(row["tick"]), row["query"], row["destination"])
                        ].append(float(row["mean_attention"]))
        x = np.array(sorted({key[0] for key in grouped}))
        budgets = {
            query: np.array(
                [
                    [np.mean(grouped[(tick, query, destination)]) for tick in x]
                    for destination in DESTINATIONS
                ]
            )
            for query in ("LTM", "STM")
        }
        tasks.append((title, x, budgets, None))

    for task, x, budgets, _ in tasks:
        assert len(x) > 1 and np.all(np.diff(x) == 1), task
        for budget in budgets.values():
            assert np.isfinite(budget).all() and np.all(budget >= 0), task
            np.testing.assert_allclose(budget.sum(axis=0), 1, atol=2e-6, err_msg=task)
    return tasks, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithmic-csv",
        type=Path,
        default=ROOT / "outputs/analysis/attention/algorithmic.csv",
    )
    parser.add_argument(
        "--visual-dir", type=Path, default=ROOT / "outputs/analysis/attention/visual"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/attention"
    )
    args = parser.parse_args()
    tasks, sources = collect_curves(args.algorithmic_csv, args.visual_dir)
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.4,
        }
    )
    palette = sns.color_palette("husl", 8)
    colors = ("#aaaaaa", palette[5], palette[0])
    fig, axes = plt.subplots(4, 2, figsize=(2.25, 2.9), sharey=True)
    plotted = []
    for row_index, (title, x, budgets, boundary) in enumerate(tasks):
        for column, query in enumerate(("LTM", "STM")):
            ax = axes[row_index, column]
            ax.stackplot(x, *budgets[query], colors=colors, alpha=0.85, linewidth=0)
            if boundary is not None:
                ax.axvline(
                    boundary, color="black", linewidth=0.5, linestyle="--", alpha=0.6
                )
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1], labels=["0", ".5", "1"])
            ticks = {
                "Copy": [0, 100, 199],
                "Sort": [0, 20, 35],
                "Maze": [1, 25, 50],
                "ImageNet-100": [1, 12, 25],
            }[title]
            ax.set_xticks(ticks)
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
            ax.set_title(
                f"{title}: {query}",
                fontsize=5.7 if title == "ImageNet-100" else 6.3,
                pad=2,
            )
            ax.tick_params(axis="both", labelsize=5.5, pad=1, length=1.5, width=0.4)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            sns.despine(ax=ax)
            for i, tick in enumerate(x):
                plotted.append(
                    {
                        "task": title,
                        "query": query,
                        "step": int(tick),
                        **{
                            destination: float(budgets[query][j, i])
                            for j, destination in enumerate(DESTINATIONS)
                        },
                    }
                )
    fig.text(0.025, 0.54, "Attention share", va="center", rotation=90, fontsize=6.5)
    shared_xlabel = fig.text(
        0.555, 0.096, "Timestep", ha="center", va="center", fontsize=6.5
    )
    fig.legend(
        [Patch(facecolor=color, alpha=0.85) for color in colors],
        DESTINATIONS,
        loc="lower center",
        ncol=3,
        fontsize=6.3,
        frameon=False,
        handlelength=1.1,
        columnspacing=1,
        handletextpad=0.4,
        bbox_to_anchor=(0.555, 0.003),
    )
    fig.subplots_adjust(
        left=0.14, right=0.97, top=0.956, bottom=0.16, hspace=0.65, wspace=0.04
    )
    # Check the compact layout before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for column in range(2):
        for row in range(3):
            upper_ticks = [
                label.get_window_extent(renderer)
                for label in axes[row, column].get_xticklabels()
            ]
            lower_title = axes[row + 1, column].title.get_window_extent(renderer)
            assert min(box.y0 for box in upper_ticks) > lower_title.y1 + 1, (
                "Overlapping panel labels"
            )
        bottom_ticks = [
            label.get_window_extent(renderer)
            for label in axes[-1, column].get_xticklabels()
        ]
        xlabel = shared_xlabel.get_window_extent(renderer)
        assert min(box.y0 for box in bottom_ticks) > xlabel.y1 + 1
        assert xlabel.y0 > fig.legends[0].get_window_extent(renderer).y1 + 1
    for row in range(4):
        left_title = axes[row, 0].title.get_window_extent(renderer)
        right_title = axes[row, 1].title.get_window_extent(renderer)
        assert left_title.x1 + 1 < right_title.x0, "Overlapping column titles"
        left_tick = axes[row, 0].get_xticklabels()[-1].get_window_extent(renderer)
        right_tick = axes[row, 1].get_xticklabels()[0].get_window_extent(renderer)
        assert left_tick.x1 + 1 < right_tick.x0, "Overlapping column tick labels"
    for ax in axes.flat:
        for label in [ax.title, *ax.get_xticklabels()]:
            box = label.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= fig.bbox.width, "Clipped panel label"
    stem = OUT / "attention_analysis"
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "sources": [str(path.resolve()) for path in sources],
                "layout": [4, 2],
                "task_order": [task[0] for task in tasks],
                "algorithmic": "Existing single-example Copy/Sort attention, averaged over queries and heads.",
                "visual": "Intact attention averaged over queries, heads, 5000 examples per seed, and seeds 0, 1, 2.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved four-by-two attention panels to {stem}", flush=True)


if __name__ == "__main__":
    main()
