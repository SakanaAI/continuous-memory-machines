"""Analyze final Maze runs.

Completed runs with metrics.json use their full-precision test curves. For
resumed runs without that file, combine the resume checkpoint metrics with
the Submitit log. The paper uses final available current-model action and
full-maze accuracies from per_seed.csv and their mean/sample SD in summary.csv.
Per-seed exports retain final epochs and their source files.
"""

import argparse
import csv
import json
import mmap
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml


from analysis.common import ROOT, CONFIGS, load_config

REPO_ROOT = ROOT
OUT_DIR = ROOT / "outputs/analysis/maze"

MODEL_ORDER = [
    "CMM",
    "CTM",
    "LSTM",
    "FSRNN",
    "RNN",
    "DNC",
    "RMC",
    "LSTM_CMM",
    "Transformer",
    "NAMTM",
    "CTMTransformer",
]
DISPLAY_NAMES = {
    "LSTM_CMM": "CMM-LSTM",
    "FSRNN": "FS-LSTM",
    "NAMTM": "NAM-TM",
    "CTMTransformer": "CMM-STM",
}
MODEL_COLORS = {
    "CMM": "#E76F7A",
    "CTM": "#4C9ED9",
    "LSTM": "#E6A85C",
    "FSRNN": "#009E73",
    "RNN": "#8E6BBE",
    "DNC": "#65B84E",
    "Transformer": "#777777",
    "RMC": "#CC79A7",
    "LSTM_CMM": "#A89B46",
    "NAMTM": "#17BECF",
    "CTMTransformer": "#D55E00",
}
METRICS = ("loss", "accuracy", "full_maze_accuracy")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TEST_PATTERN = re.compile(
    rf"^Test Epoch (?P<epoch>\d+) current: "
    rf"loss=(?P<loss>{FLOAT}), "
    rf"accuracy=(?P<accuracy>{FLOAT}), "
    rf"full_maze_accuracy=(?P<full_maze_accuracy>{FLOAT}),"
)

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)


def load_yaml(path):
    with path.open() as handle:
        return yaml.safe_load(handle)


def numeric_run_dirs(base_dir):
    return sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def submitit_logs(base_dir):
    indexed = {}
    for path in (base_dir / ".submitit").glob("*_*/*_log.out"):
        match = re.search(r"_(\d+)$", path.parent.name)
        if match:
            indexed[int(match.group(1))] = path
    return indexed


def checkpoint_path(cfg, seed):
    raw_path = str(cfg["checkpoint"]["path"]).replace("${seed}", str(seed))
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_curves(path):
    """Read only the test curves; training attention histories can be very large."""
    curves = {}
    decoder = json.JSONDecoder()
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for metric in METRICS:
            key = f'"test_{metric}"'.encode()
            match = re.search(re.escape(key) + rb"\s*:\s*", mapped)
            if match is None:
                raise KeyError(f"Missing {key.decode()} in {path}")
            start, size = match.end(), 1024 * 1024
            while True:
                end = min(start + size, len(mapped))
                try:
                    values, _ = decoder.raw_decode(mapped[start:end].decode())
                    break
                except json.JSONDecodeError:
                    if end == len(mapped):
                        raise
                    size *= 2
            curves[metric] = {int(epoch): float(value) for epoch, value in values}
    return curves


def load_checkpoint_curves(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    logged = checkpoint["metrics"]
    curves = {}
    for metric in METRICS:
        key = f"test_{metric}"
        curves[metric] = {int(epoch): float(value) for epoch, value in logged[key]}
    return curves, path


def checkpoint_curves(cfg, seed):
    path = checkpoint_path(cfg, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {path}")
    return load_checkpoint_curves(path)


def latest_checkpoint_curves(run_dir):
    checkpoints = {}
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", path.name)
        if match and path.is_file():
            checkpoints[int(match.group(1))] = path
    if not checkpoints:
        raise FileNotFoundError(f"No saved epoch checkpoints in {run_dir}")
    epoch = max(checkpoints)
    curves, path = load_checkpoint_curves(checkpoints[epoch])
    if any(not values or max(values) != epoch for values in curves.values()):
        raise RuntimeError(
            f"Checkpoint test metrics do not end at epoch {epoch}: {path}"
        )
    return curves, path, epoch


def continuation_curves(log_path, require_completed=True):
    curves = {metric: {} for metric in METRICS}
    completed = False
    with log_path.open(errors="replace") as handle:
        for line in handle:
            match = TEST_PATTERN.match(line)
            if match:
                epoch = int(match.group("epoch"))
                for metric in METRICS:
                    curves[metric][epoch] = float(match.group(metric))
            if "Job completed successfully" in line:
                completed = True
    if require_completed and not completed:
        raise RuntimeError(
            f"Submitit log does not report successful completion: {log_path}"
        )
    return curves


def collect_runs(config, final_epochs=None, log_dirs=None, latest_checkpoints=None):
    final_epochs = final_epochs or {}
    log_dirs = log_dirs or {}
    latest_checkpoints = set(latest_checkpoints or [])
    runs = defaultdict(list)
    curve_rows = []
    per_seed_rows = []

    for model in MODEL_ORDER:
        base_dir = REPO_ROOT / config[model]
        run_dirs = numeric_run_dirs(base_dir)
        logs = submitit_logs(REPO_ROOT / log_dirs.get(model, config[model]))
        use_checkpoint = model in latest_checkpoints
        if len(run_dirs) != 3:
            raise RuntimeError(
                f"Expected three run directories for {model}; "
                f"found {len(run_dirs)} runs and {len(logs)} logs"
            )

        for array_index, run_dir in enumerate(run_dirs):
            cfg = load_yaml(run_dir / ".hydra/config.yaml")
            seed = int(cfg["seed"])
            log_path = logs.get(array_index)
            log_curves = None
            metrics_path = run_dir / "metrics.json"
            ckpt_path = None
            metrics_file = ""
            continuation_logs = []
            expected_epoch = final_epochs.get(model, {}).get(seed, 9000)
            if use_checkpoint:
                curves, ckpt_path, expected_epoch = latest_checkpoint_curves(run_dir)
                for continuation in sorted(
                    run_dir.glob("wandb/run-*/files/output.log")
                ):
                    extra = continuation_curves(continuation, require_completed=False)
                    for metric in METRICS:
                        for epoch, value in extra[metric].items():
                            # Checkpoint values take precedence over rounded log values.
                            curves[metric].setdefault(epoch, value)
                    continuation_logs.append(str(continuation.resolve()))
                expected_epoch = max(curves["full_maze_accuracy"])
            elif metrics_path.is_file():
                metrics_file = str(metrics_path.resolve())
                curves = load_json_curves(metrics_path)
            else:
                if log_path is None:
                    raise RuntimeError(
                        f"Missing metrics and continuation log: {run_dir}"
                    )
                log_curves = continuation_curves(log_path)
                first_half, ckpt_path = checkpoint_curves(cfg, seed)
                original_metrics = ckpt_path.parent / "metrics.json"
                if original_metrics.is_file():
                    metrics_file = str(original_metrics.resolve())
                    precise_curves = load_json_curves(original_metrics)
                    for metric in METRICS:
                        first_half[metric].update(precise_curves[metric])
                curves = {}
                for metric in METRICS:
                    combined = dict(log_curves[metric])
                    combined.update(first_half[metric])
                    curves[metric] = combined

            common_epochs = sorted(set.intersection(*(set(curves[m]) for m in METRICS)))
            if not common_epochs or common_epochs[-1] != expected_epoch:
                raise RuntimeError(
                    f"Incomplete curve for {model} seed {seed}: "
                    f"expected final epoch {expected_epoch}"
                )

            run = {"seed": seed, "curves": curves}
            runs[model].append(run)
            for epoch in common_epochs:
                curve_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "epoch": epoch,
                        "test_loss": curves["loss"][epoch],
                        "test_action_accuracy": curves["accuracy"][epoch],
                        "test_full_maze_accuracy": curves["full_maze_accuracy"][epoch],
                    }
                )

            final_epoch = common_epochs[-1]
            per_seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "final_epoch": final_epoch,
                    "continuation_logs": ";".join(continuation_logs),
                    "test_loss": curves["loss"][final_epoch],
                    "test_action_accuracy": curves["accuracy"][final_epoch],
                    "test_full_maze_accuracy": curves["full_maze_accuracy"][
                        final_epoch
                    ],
                    "checkpoint": str(ckpt_path.resolve()) if ckpt_path else "",
                    "metrics_file": metrics_file,
                    "submitit_log": str(log_path.resolve()) if log_path else "",
                }
            )
    return runs, curve_rows, per_seed_rows


def aggregate(per_seed_rows):
    rows = []
    for model in MODEL_ORDER:
        model_rows = [row for row in per_seed_rows if row["model"] == model]
        output = {
            "model": model,
            "n_seeds": len(model_rows),
            "min_final_epoch": min(row["final_epoch"] for row in model_rows),
            "max_final_epoch": max(row["final_epoch"] for row in model_rows),
        }
        for metric in (
            "test_loss",
            "test_action_accuracy",
            "test_full_maze_accuracy",
        ):
            values = np.asarray([row[metric] for row in model_rows], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            output[f"mean_{metric}"] = mean
            output[f"std_{metric}"] = std
        rows.append(output)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def aligned_metric(model_runs, metric):
    common_epochs = sorted(
        set.intersection(*(set(run["curves"][metric]) for run in model_runs))
    )
    values = np.asarray(
        [
            [run["curves"][metric][epoch] for epoch in common_epochs]
            for run in model_runs
        ]
    )
    return np.asarray(common_epochs), values.mean(axis=0), values.std(axis=0, ddof=1)


def plot_metric(runs, metric, ylabel, stem, ylim=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for model in MODEL_ORDER:
        epochs, mean, std = aligned_metric(runs[model], metric)
        color = MODEL_COLORS[model]
        ax.plot(
            epochs,
            mean,
            color=color,
            linewidth=1.7,
            label=DISPLAY_NAMES.get(model, model),
        )
        ax.fill_between(
            epochs,
            np.maximum(mean - std, 0),
            mean + std,
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(
        0,
        max(
            9000,
            max(
                max(run["curves"][metric])
                for model_runs in runs.values()
                for run in model_runs
            ),
        ),
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT_DIR, REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "maze.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/maze"
    )
    args = parser.parse_args()
    REPO_ROOT = args.runs_root.resolve()
    OUT_DIR = args.output_dir
    config = load_config(args.config, REPO_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, curve_rows, per_seed_rows = collect_runs(
        config["models"],
        config.get("final_epochs", {}),
        config.get("log_dirs", {}),
        config.get("latest_checkpoints", []),
    )
    summary_rows = aggregate(per_seed_rows)

    write_csv(OUT_DIR / "curves.csv", curve_rows)
    write_csv(OUT_DIR / "per_seed.csv", per_seed_rows)
    write_csv(OUT_DIR / "summary.csv", summary_rows)

    plot_metric(runs, "loss", "Test Loss", "test_loss")
    plot_metric(
        runs, "accuracy", "Test Action Accuracy", "test_action_accuracy", (0, 1.01)
    )
    plot_metric(
        runs,
        "full_maze_accuracy",
        "Test Full-Maze Accuracy",
        "test_full_maze_accuracy",
        (0, 1.01),
    )
    for row in summary_rows:
        print(
            f"{row['model']}: action={100 * row['mean_test_action_accuracy']:.2f}%, "
            f"full maze={100 * row['mean_test_full_maze_accuracy']:.2f}%"
        )
    print(f"\nSaved Maze tables and plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
