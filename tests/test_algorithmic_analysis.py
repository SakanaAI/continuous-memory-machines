import csv
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from analysis.algorithmic import plot_results, select_results
from analysis.common import load_config
from analysis.results import collect_results, summarize


def make_run(base, job, seed, epoch, metrics, task=None):
    run = base / str(job)
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra/config.yaml").write_text(
        json.dumps(
            {
                "seed": seed,
                "training": {"epochs": epoch},
                "task": task or {},
            }
        )
    )
    (run / "summary.json").write_text("{}")
    (run / "metrics.json").write_text(
        json.dumps({"train_loss": [[epoch, 0.1]], **metrics})
    )
    return run


def test_paper_results_select_final_epoch_and_actual_seeds(tmp_path):
    base = tmp_path / "runs"
    for job, seed, loss in ((0, 0, 0.2), (1, 1, 0.4), (2, 2, 0.9)):
        make_run(
            base,
            job,
            seed,
            1000,
            {"test_loss": [[500, 0.01], [1000, loss], [1100, 0.8]]},
        )
    config = {"copy": {"models": {"GDNStateTracking": str(base)}}}
    rows = collect_results(config)
    assert [row["value"] for row in rows] == [0.2, 0.4, 0.9]
    assert {row["model"] for row in rows} == {"GDN"}
    assert {row["source_model"] for row in rows} == {"GDNStateTracking"}
    summary = summarize(rows)[0]
    assert summary["n_seeds"] == 3
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["std"] == pytest.approx(np.std([0.2, 0.4, 0.9], ddof=1))


def test_duplicate_actual_seeds_are_rejected(tmp_path):
    for job in (0, 1):
        make_run(tmp_path, job, 0, 1000, {"test_loss": [[1000, 0.1]]})
    with pytest.raises(ValueError, match="Duplicate training seed"):
        collect_results({"copy": {"models": {"CMM": str(tmp_path)}}})


def test_incomplete_parity_is_skipped_and_all_reported_lengths_are_retained(tmp_path):
    for seed, epoch in ((0, 200), (1, 150)):
        metrics = {
            f"test_{metric}_len_{length}": [[epoch, 0.1]]
            for metric in ("loss", "accuracy")
            for length in (40, 1000)
        }
        make_run(tmp_path, seed, seed, epoch, metrics, {"test_seq_lengths": [40, 1000]})
    with pytest.warns(UserWarning):
        rows = collect_results({"parity": {"models": {"CMM": str(tmp_path)}}})
    assert {row["seed"] for row in rows} == {0}
    assert len(rows) == 4
    assert {row["eval_length"] for row in rows} == {40, 1000}


def test_missing_query_accuracy_is_not_inferred_from_regression_loss(tmp_path):
    for seed in range(3):
        make_run(tmp_path, seed, seed, 1000, {"test_loss": [[1000, 0.002]]})
    with pytest.warns(
        UserWarning, match="Missing or ambiguous final test_query_accuracy"
    ):
        rows = collect_results(
            {"regression_linear_k20": {"models": {"CMM": str(tmp_path)}}}
        )
    assert {row["metric"] for row in rows} == {"loss"}


def test_manifests_resolve_run_paths_without_changing_labels(tmp_path):
    path = tmp_path / "runs.yaml"
    path.write_text("models:\n  CMM: [outputs/Copy/CMM, /archive/seed]\nmetric: loss\n")
    config = load_config(path, tmp_path)
    assert config == {
        "models": {"CMM": [str(tmp_path / "outputs/Copy/CMM"), "/archive/seed"]},
        "metric": "loss",
    }


def test_combined_uses_parity_boundary_and_preserves_missing_models(tmp_path):
    rows = []
    for task in ("copy", "associative_recall", "priority_sort", "parity"):
        for model in ("CMM", "LSTM"):
            if task == "parity" and model == "LSTM":
                continue
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "source_model": model,
                    "metric": "loss",
                    "n_seeds": 3,
                    "mean": 1e-9 if task == "parity" else 0.1,
                    "std": 0.0,
                    "eval_length": 100
                    if task == "copy"
                    else 40
                    if task == "parity"
                    else "",
                }
            )
    rows.append({**rows[-1], "eval_length": 1000, "mean": 0.9})
    data = select_results(rows)
    assert data["parity"]["CMM"]["mean"] == 1e-9
    assert "LSTM" not in data["parity"]
    plot_results(data, tmp_path)
    with (tmp_path / "summary.csv").open() as handle:
        plotted = list(csv.DictReader(handle))
    assert len(plotted) == 7
    assert [row["eval_length"] for row in plotted if row["task"] == "parity"] == ["40"]
    for suffix in ("", "_log"):
        for extension in ("pdf", "png"):
            assert (tmp_path / f"combined_bar{suffix}.{extension}").stat().st_size > 100
