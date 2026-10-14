"""Evaluate Priority Sort checkpoints within and beyond their training lengths."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis.common import ROOT, CONFIGS, load_config
from tasks import construct_task
from tasks.recall.priority_sort import PrioritySortDataset
from utils import load_cfg_from_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path("outputs/analysis/priority_sort/length_generalization")
MODEL_ORDER = ["CMM", "CTM_Matched", "Transformer", "RMT", "DNC"]
DISPLAY_NAMES = {
    "CMM": "CMM",
    "CTM_Matched": "CTM-Ext.",
    "Transformer": "Transformer",
    "RMT": "RMT",
    "DNC": "DNC",
}
COLORS = {
    "CMM": "#E76F7A",
    "CTM_Matched": "#4C9ED9",
    "Transformer": "0.6",
    "RMT": "#E69F00",
    "DNC": "#65B84E",
}

sns.set_palette("hls")
sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[4, 8, 12, 16, 20, 30, 40, 50]
    )
    parser.add_argument("--num-samples", type=int, default=832)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", type=int, default=-1, help="CUDA device index; -1 uses CPU."
    )
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Models to evaluate. By default, evaluate all models.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Keep existing rows outside the selected model/length pairs.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from the existing summary CSV without evaluating checkpoints.",
    )
    parser.add_argument("--config", type=Path, default=CONFIGS / "priority_sort.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/priority_sort/generalization",
    )
    return parser.parse_args()


def completed_seed_dirs(base_dir):
    path = Path(base_dir)
    return sorted(
        [child for child in path.iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda child: int(child.name),
    )


def final_checkpoint(seed_dir):
    checkpoints = list(seed_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None

    def epoch(path):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=epoch)


def load_task(checkpoint_path, device):
    cfg, task_type, model_type = load_cfg_from_checkpoint(str(checkpoint_path))
    with open_dict(cfg):
        cfg.training.compile = False
        cfg.device = device.index if device.type == "cuda" else -1

    task = construct_task(cfg, task_type=task_type)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key.removeprefix("_orig_mod.").replace("start_trace", "start_stm"): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    task.model.load_state_dict(state_dict)
    task.model.to(device)
    task.model.eval()
    return task, cfg, model_type


def evaluate(task, cfg, num_input_items, num_samples, batch_size, eval_seed, device):
    dataset = PrioritySortDataset(
        value_width=cfg.task.value_width,
        num_input_items=num_input_items,
        num_output_items=cfg.task.num_output_items,
        num_samples=num_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Reset before every evaluation so all checkpoints see identical examples.
    torch.manual_seed(eval_seed + num_input_items)
    accumulator = defaultdict(list)
    for inputs, targets in dataloader:
        with torch.inference_mode():
            inputs, targets = inputs.to(device), targets.to(device)
            states = task.model.get_initial_states(batch_size=inputs.size(0))
            logits, _ = task.model(inputs, states=states)
            loss = task._compute_loss(logits, targets)
            metrics = task._compute_metrics(logits, targets, loss)
        for key, value in metrics.items():
            accumulator[key].append(value)
    return {key: float(np.mean(values)) for key, values in accumulator.items()}


def aggregate(per_seed_rows):
    grouped = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["model"], row["eval_num_input_items"])].append(row)

    summary_rows = []
    for model in MODEL_ORDER:
        model_lengths = sorted(
            length for candidate, length in grouped if candidate == model
        )
        for length in model_lengths:
            rows = grouped[(model, length)]
            output = {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "eval_num_input_items": length,
                "total_sequence_steps": rows[0]["total_sequence_steps"],
                "n_seeds": len(rows),
            }
            for metric in ("loss", "bit_accuracy", "vector_accuracy"):
                values = np.asarray([row[metric] for row in rows], dtype=float)
                output[f"mean_{metric}"] = values.mean()
                output[f"std_{metric}"] = values.std(ddof=1) if len(values) > 1 else 0.0
            summary_rows.append(output)
    return summary_rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path):
    integer_fields = {"eval_num_input_items", "total_sequence_steps", "n_seeds"}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key in integer_fields:
                row[key] = int(value)
            elif key.startswith(("mean_", "std_")):
                row[key] = float(value)
    return rows


def read_per_seed_csv(path):
    integer_fields = {
        "seed",
        "trained_min_num_input_items",
        "trained_max_num_input_items",
        "eval_num_input_items",
        "total_sequence_steps",
        "num_samples",
    }
    float_fields = {"loss", "bit_accuracy", "vector_accuracy"}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in integer_fields:
            row[key] = int(row[key])
        for key in float_fields:
            row[key] = float(row[key])
    return rows


def mark_training_cutoff(ax):
    ax.axvline(16, color="0.42", linestyle="--", linewidth=1.2, zorder=0)
    ax.margins(x=0.03)


def model_legend_handles(models):
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label=DISPLAY_NAMES[model],
        )
        for model in models
    ] + [
        Line2D(
            [0],
            [0],
            color="0.42",
            linestyle="--",
            linewidth=1.2,
            label="Training Phase",
        )
    ]


def plot_loss(summary_rows, models, suffix):
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    for model in models:
        rows = [row for row in summary_rows if row["model"] == model]
        lengths = np.asarray([row["eval_num_input_items"] for row in rows])
        means = np.asarray([row["mean_loss"] for row in rows])
        stds = np.asarray([row["std_loss"] for row in rows])
        ax.errorbar(
            lengths,
            means,
            yerr=stds,
            marker="o",
            markersize=4.5,
            capsize=2.5,
            elinewidth=1,
            linewidth=1.8,
            color=COLORS[model],
            alpha=0.9,
            label=DISPLAY_NAMES[model],
        )
    mark_training_cutoff(ax)
    ax.set_xlabel("Number of Input Items ($N$)")
    ax.set_ylabel("Test Loss")
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted({row["eval_num_input_items"] for row in summary_rows}))
    ax.grid(True, alpha=0.3)
    ax.legend(handles=model_legend_handles(models), ncol=2, loc="upper left")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"extrapolation_loss{suffix}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_accuracy(summary_rows, models, suffix, metric, ylabel, stem):
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    for model in models:
        rows = [row for row in summary_rows if row["model"] == model]
        lengths = np.asarray([row["eval_num_input_items"] for row in rows])
        means = np.asarray([row[f"mean_{metric}"] for row in rows])
        stds = np.asarray([row[f"std_{metric}"] for row in rows])
        ax.errorbar(
            lengths,
            means,
            yerr=stds,
            marker="o",
            markersize=4.5,
            capsize=2.5,
            elinewidth=0.9,
            linewidth=1.8,
            linestyle="-",
            color=COLORS[model],
            alpha=0.9,
            label=DISPLAY_NAMES[model],
        )
    mark_training_cutoff(ax)
    ax.set_xlabel("Number of Input Items ($N$)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(sorted({row["eval_num_input_items"] for row in summary_rows}))
    ax.grid(True, alpha=0.3)
    ax.legend(handles=model_legend_handles(models), ncol=2, loc="lower left")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"{stem}{suffix}.{extension}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def plot_accuracies(summary_rows, models, suffix):
    plot_accuracy(
        summary_rows,
        models=models,
        suffix=suffix,
        metric="bit_accuracy",
        ylabel="Test Bit Accuracy",
        stem="extrapolation_bit_accuracy",
    )
    plot_accuracy(
        summary_rows,
        models=models,
        suffix=suffix,
        metric="vector_accuracy",
        ylabel="Test Vector Accuracy",
        stem="extrapolation_vector_accuracy",
    )


def plot_all(summary_rows):
    available_models = {row["model"] for row in summary_rows}
    without_dnc = [model for model in MODEL_ORDER if model != "DNC"]
    for candidates, suffix in ((without_dnc, ""), (MODEL_ORDER, "_with_dnc")):
        models = [model for model in candidates if model in available_models]
        if not models:
            continue
        plot_loss(summary_rows, models, suffix)
        plot_accuracies(summary_rows, models, suffix)


def main():
    global OUT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    if args.plot_only:
        summary_path = OUT_DIR / "extrapolation_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Cannot use --plot-only; missing {summary_path}")
        summary_rows = read_summary_csv(summary_path)
        plot_all(summary_rows)
        print(f"Regenerated checkpoint extrapolation plots in {OUT_DIR}")
        return
    if args.device >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is unavailable")
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    experiment_dirs = load_config(args.config, args.runs_root)["length_generalization"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_path = OUT_DIR / "extrapolation_per_seed.csv"
    per_seed_rows = []
    if args.reuse_existing:
        if not per_seed_path.is_file():
            raise FileNotFoundError(
                f"Cannot reuse existing rows; missing {per_seed_path}"
            )
        existing_rows = read_per_seed_csv(per_seed_path)
        requested_lengths = set(args.lengths)
        per_seed_rows = [
            row
            for row in existing_rows
            if row["model"] not in args.models
            or row["eval_num_input_items"] not in requested_lengths
        ]
    print(f"Device: {device}; lengths={args.lengths}; samples={args.num_samples}")
    for model in args.models:
        seed_dirs = completed_seed_dirs(experiment_dirs[model])
        for seed_dir in seed_dirs:
            checkpoint_path = final_checkpoint(seed_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
            task, cfg, checkpoint_model_type = load_task(checkpoint_path, device)
            print(f"{model} seed={cfg.seed}: {checkpoint_path.name}")
            for length in tqdm(args.lengths, desc=f"{model} seed {cfg.seed}"):
                metrics = evaluate(
                    task,
                    cfg,
                    length,
                    args.num_samples,
                    args.batch_size,
                    args.eval_seed,
                    device,
                )
                per_seed_rows.append(
                    {
                        "model": model,
                        "display_name": DISPLAY_NAMES[model],
                        "checkpoint_model_type": checkpoint_model_type,
                        "seed": cfg.seed,
                        "trained_min_num_input_items": cfg.task.train_min_num_input_items,
                        "trained_max_num_input_items": cfg.task.train_max_num_input_items,
                        "eval_num_input_items": length,
                        "total_sequence_steps": length + cfg.task.num_output_items,
                        "num_samples": args.num_samples,
                        "loss": metrics["loss"],
                        "bit_accuracy": metrics["bit_accuracy"],
                        "vector_accuracy": metrics["vector_accuracy"],
                        "checkpoint": str(checkpoint_path),
                    }
                )
            del task
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = aggregate(per_seed_rows)
    write_csv(per_seed_path, per_seed_rows)
    write_csv(OUT_DIR / "extrapolation_summary.csv", summary_rows)
    plot_all(summary_rows)
    print(f"Saved checkpoint extrapolation results to {OUT_DIR}")


if __name__ == "__main__":
    main()
