"""Plot the paper's four-task algorithmic comparison from final result summaries."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.common import ROOT, MODEL_ORDER, DISPLAY_NAMES, write_csv

TASKS = {
    "copy": "Copy",
    "associative_recall": "Recall",
    "priority_sort": "Sort",
    "parity": "Parity",
}


def select_results(rows):
    data = {task: {} for task in TASKS}
    for row in rows:
        task, model = row["task"], row["model"]
        if task not in data or model not in MODEL_ORDER or row["metric"] != "loss":
            continue
        if task in {"copy", "parity"} and int(row["eval_length"]) != (
            100 if task == "copy" else 40
        ):
            continue
        if model in data[task]:
            raise ValueError(f"Multiple final loss summaries for {task}/{model}")
        data[task][model] = {
            **row,
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    if not any(data.values()):
        raise ValueError("No algorithmic losses in summary")
    return data


def plot_results(data, output_dir):
    output_dir = Path(output_dir)
    sns.set_style("whitegrid")
    models = [
        model for model in MODEL_ORDER if any(model in rows for rows in data.values())
    ]
    write_csv(
        output_dir / "summary.csv",
        [rows[model] for rows in data.values() for model in models if model in rows],
    )
    colors = [*sns.color_palette("husl", 3), sns.color_palette("husl", 4)[3]]
    width, x = 0.2, np.arange(len(models))
    for scale, suffix in (("linear", ""), ("log", "_log")):
        fig, ax = plt.subplots(figsize=(max(14, len(models)), 4.5))
        for index, (task, title) in enumerate(TASKS.items()):
            rows = data[task]
            present = np.array(
                [i for i, model in enumerate(models) if model in rows], dtype=int
            )
            ax.bar(
                x[present] + index * width,
                [rows[models[i]]["mean"] for i in present],
                width,
                yerr=[rows[models[i]]["std"] for i in present],
                capsize=3,
                color=colors[index],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=title,
            )
            for i, model in enumerate(models):
                if model not in rows:
                    ax.text(
                        x[i] + index * width,
                        0.025,
                        "N/A",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=colors[index],
                        transform=ax.get_xaxis_transform(),
                    )
        ax.set_yscale(scale)
        if scale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_xticks(
            x + 1.5 * width, [DISPLAY_NAMES[model] for model in models], fontsize=12
        )
        for model, label in zip(models, ax.get_xticklabels()):
            if model in {"Transformer", "LSTM_CMM", "CTM_Matched", "CTMTransformer"}:
                label.set_color("0.45")
        ax.set_xlabel("Model", fontsize=19, labelpad=10)
        ax.set_ylabel("Test Loss", fontsize=19, labelpad=12)
        ax.legend(fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        sns.despine(ax=ax)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output_dir / f"combined_bar{suffix}.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/analysis/results/summary.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/algorithmic"
    )
    args = parser.parse_args()
    with args.summary.open() as handle:
        data = select_results(list(csv.DictReader(handle)))
    plot_results(data, args.output_dir)
    print(f"Saved algorithmic figures to {args.output_dir}")


if __name__ == "__main__":
    main()
