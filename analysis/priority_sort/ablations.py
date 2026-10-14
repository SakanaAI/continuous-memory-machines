"""Priority Sort LTM-capacity and memory-Transformer-depth ablations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from analysis.common import CONFIGS, ROOT, load_config, seed_dirs, write_csv


def collect_runs(config):
    rows = []
    seen = set()
    for ablation in ("transformer_depth", "ltm_tokens_ms5"):
        sweep = config[ablation]
        for run in seed_dirs(sweep["dir"]):
            required = [
                run / "metrics.json",
                run / ".hydra/config.yaml",
                run / "summary.json",
            ]
            if not all(path.is_file() for path in required):
                warnings.warn(f"Skipping incomplete run: {run}")
                continue
            metrics = json.loads(required[0].read_text())
            cfg = yaml.safe_load(required[1].read_text())
            if cfg["model"]["model"] != "CMM":
                raise ValueError(f"Expected CMM ablation: {run}")
            seed, value = int(cfg["seed"]), int(cfg["model"][sweep["parameter"]])
            if seed not in config["seeds"] or value not in sweep["values"]:
                continue
            if ablation == "ltm_tokens_ms5" and cfg["model"]["memory_length"] != 5:
                raise ValueError(f"The capacity ablation requires STM length 5: {run}")
            values = [
                value for epoch, value in metrics.get("test_loss", []) if epoch == 1000
            ]
            if len(values) != 1:
                warnings.warn(f"Skipping incomplete run: {run}")
                continue
            key = ablation, value, seed
            if key in seen:
                raise ValueError(f"Duplicate sweep seed {key}: {run}")
            seen.add(key)
            loss = float(values[0])
            if not np.isfinite(loss):
                raise ValueError(f"Nonfinite test loss: {run}")
            rows.append(
                {
                    "ablation": ablation,
                    "sweep_value": value,
                    "seed": seed,
                    "epoch": 1000,
                    "test_loss": loss,
                    "run_dir": str(run),
                }
            )
        missing = {
            (ablation, value, seed)
            for value in sweep["values"]
            for seed in config["seeds"]
        } - seen
        if missing:
            warnings.warn(f"Missing final ablation runs: {sorted(missing)}")
    if not rows:
        raise RuntimeError(
            "No completed ablations found; set --runs-root or edit --config"
        )
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["ablation"], row["sweep_value"]].append(row["test_loss"])
    return [
        {
            "ablation": ablation,
            "sweep_value": value,
            "n_seeds": len(losses),
            "mean_test_loss": float(np.mean(losses)),
            "std_test_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        }
        for (ablation, value), losses in sorted(groups.items())
    ]


def plot_results(rows, output_dir):
    sns.set_style("whitegrid")
    for ablation, xlabel, stem in (
        (
            "transformer_depth",
            r"Transformer Layers ($J$)",
            "priority_sort_transformer_depth",
        ),
        ("ltm_tokens_ms5", r"LTM Tokens ($M_L$)", "priority_sort_ltm_tokens_ms5"),
    ):
        selected = [row for row in rows if row["ablation"] == ablation]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        x = [row["sweep_value"] for row in selected]
        ax.errorbar(
            x,
            [row["mean_test_loss"] for row in selected],
            yerr=[row["std_test_loss"] for row in selected],
            marker="o",
            capsize=4,
            color="#E76F7A",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(value) for value in x])
        ax.set(xlabel=xlabel, ylabel="Test Loss", ylim=(0, None))
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "priority_sort.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/priority_sort/ablations",
    )
    args = parser.parse_args()
    rows = collect_runs(load_config(args.config, args.runs_root)["ablations"])
    summary = aggregate(rows)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    plot_results(summary, args.output_dir)
    print(f"Saved ablation tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
