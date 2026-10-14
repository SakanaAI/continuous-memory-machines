"""Summarize final algorithmic and regression results used in the paper."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import numpy as np
import yaml

from analysis.common import (
    CONFIGS,
    ROOT,
    DISPLAY_NAMES,
    load_config,
    seed_dirs,
    write_csv,
)


def metric_specs(task, cfg):
    if task == "parity":
        return [
            (f"test_{metric}_len_{length}", metric, length)
            for length in cfg["task"]["test_seq_lengths"]
            for metric in ("loss", "accuracy")
        ]
    if task.startswith("regression_"):
        return [
            ("test_loss", "loss", ""),
            ("test_query_accuracy", "query_accuracy", ""),
        ]
    specs = [("test_loss", "loss", 100 if task == "copy" else "")]
    if task in {"associative_recall", "priority_sort"}:
        specs += [
            (f"test_{metric}", metric, "")
            for metric in ("bit_accuracy", "vector_accuracy")
        ]
    return specs


def collect_results(config):
    """Select final epochs, identify actual training seeds, and reject duplicates."""
    rows = []
    seen = set()
    for task, experiment in config.items():
        final_epoch = 200 if task == "parity" else 1000
        for source_model, directories in experiment["models"].items():
            model = "GDN" if source_model == "GDNStateTracking" else source_model
            if isinstance(directories, str):
                directories = [directories]
            completed = set()
            for directory in directories:
                for run in seed_dirs(directory):
                    required = [
                        run / "metrics.json",
                        run / "summary.json",
                        run / ".hydra/config.yaml",
                    ]
                    if not all(path.is_file() for path in required):
                        warnings.warn(f"Skipping incomplete run: {run}")
                        continue
                    metrics = json.loads(required[0].read_text())
                    cfg = yaml.safe_load(required[2].read_text())
                    seed = int(cfg["seed"])
                    if (
                        not metrics.get("train_loss")
                        or metrics["train_loss"][-1][0] < final_epoch
                    ):
                        warnings.warn(f"Skipping incomplete run: {run}")
                        continue
                    key = (task, model, seed)
                    if key in seen:
                        raise ValueError(f"Duplicate training seed {key}: {run}")
                    seen.add(key)
                    for name, metric, length in metric_specs(task, cfg):
                        points = [
                            value
                            for epoch, value in metrics.get(name, [])
                            if epoch == final_epoch
                        ]
                        if len(points) != 1:
                            warnings.warn(
                                f"Missing or ambiguous final {name} at epoch {final_epoch}: {run}"
                            )
                            continue
                        value = float(points[0])
                        if not np.isfinite(value):
                            raise ValueError(f"Nonfinite {name}: {run}")
                        if metric == "loss":
                            completed.add(seed)
                        rows.append(
                            {
                                "task": task,
                                "model": model,
                                "source_model": source_model,
                                "seed": seed,
                                "epoch": final_epoch,
                                "metric": metric,
                                "eval_length": length,
                                "value": value,
                                "run_dir": str(run),
                            }
                        )
            missing = sorted(set(experiment.get("seeds", [0, 1, 2])) - completed)
            if missing:
                warnings.warn(
                    f"Missing completed {task}/{source_model} seeds: {missing}"
                )
    if not rows:
        raise RuntimeError(
            "No completed results found; set --runs-root or edit --config to point to your runs"
        )
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["task"],
                row["model"],
                row["source_model"],
                row["metric"],
                row["eval_length"],
            )
        ].append(row)
    summaries = []
    for (task, model, source, metric, length), records in groups.items():
        if len({row["seed"] for row in records}) != len(records):
            raise ValueError(
                f"Duplicate training seeds for {task}/{model}/{metric}/{length}"
            )
        values = np.array([row["value"] for row in records], dtype=float)
        summaries.append(
            {
                "task": task,
                "model": model,
                "source_model": source,
                "display_name": DISPLAY_NAMES[model],
                "metric": metric,
                "eval_length": length,
                "n_seeds": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "results.yaml")
    parser.add_argument(
        "--runs-root", type=Path, default=ROOT, help="Directory containing outputs/"
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Only summarize these keys from the manifest"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/results"
    )
    args = parser.parse_args()
    config = load_config(args.config, args.runs_root)
    if args.tasks:
        unknown = set(args.tasks) - config.keys()
        if unknown:
            parser.error(f"Unknown tasks: {', '.join(sorted(unknown))}")
        config = {task: config[task] for task in args.tasks}
    rows = collect_results(config)
    summary = summarize(rows)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    regression = [
        {
            "task_mode": row["task"].removeprefix("regression_"),
            "model": row["model"],
            "source_model": row["source_model"],
            "n_seeds": row["n_seeds"],
            "mean_accuracy_percent": 100 * row["mean"],
            "std_accuracy_pp": 100 * row["std"],
        }
        for row in summary
        if row["metric"] == "query_accuracy"
    ]
    if regression:
        write_csv(args.output_dir / "regression_table.csv", regression)
    print(f"Saved {len(rows)} per-seed measurements to {args.output_dir}")


if __name__ == "__main__":
    main()
