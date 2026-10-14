import json

import pytest
import torch

from analysis.maze import results as analysis


def make_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(analysis, "MODEL_ORDER", ["NAMTM"])
    for seed, epoch in enumerate((10000, 8500, 8000)):
        run = tmp_path / "runs" / str(seed)
        (run / ".hydra").mkdir(parents=True)
        (run / ".hydra/config.yaml").write_text(json.dumps({"seed": seed}))
        for saved_epoch in (500, epoch):
            metrics = {
                f"test_{metric}": [(saved_epoch, (seed + 1) / 10)]
                for metric in analysis.METRICS
            }
            torch.save({"metrics": metrics}, run / f"checkpoint_epoch_{saved_epoch}.pt")
        # Existing metrics must not override an explicitly requested checkpoint.
        (run / "metrics.json").write_text("{}")
    # Lexicographic sorting would incorrectly select 9000 instead of 10000.
    torch.save({"metrics": {}}, tmp_path / "runs/0/checkpoint_epoch_9000.pt")
    return {"NAMTM": "runs"}


def test_latest_checkpoints_include_unfinished_runs_and_preserve_epochs(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    runs, curves, per_seed = analysis.collect_runs(config, latest_checkpoints=["NAMTM"])
    assert len(runs["NAMTM"]) == len(curves) == len(per_seed) == 3
    assert [row["final_epoch"] for row in per_seed] == [10000, 8500, 8000]
    assert [row["test_full_maze_accuracy"] for row in per_seed] == [0.1, 0.2, 0.3]
    for row in per_seed:
        assert row["checkpoint"].endswith(f"checkpoint_epoch_{row['final_epoch']}.pt")
        assert row["metrics_file"] == row["submitit_log"] == ""
    summary = analysis.aggregate(per_seed)[0]
    assert summary["n_seeds"] == 3
    assert summary["min_final_epoch"] == 8000
    assert summary["max_final_epoch"] == 10000
    assert summary["mean_test_full_maze_accuracy"] == pytest.approx(0.2)
    assert summary["std_test_full_maze_accuracy"] == pytest.approx(0.1)


def test_unfinished_runs_require_checkpoint_opt_in(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    with pytest.raises(KeyError, match="Missing"):
        analysis.collect_runs(config)


def test_latest_checkpoint_rejects_metrics_from_an_earlier_epoch(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    torch.save(
        {"metrics": {f"test_{metric}": [(9995, 0.1)] for metric in analysis.METRICS}},
        tmp_path / "runs/0/checkpoint_epoch_10000.pt",
    )
    with pytest.raises(RuntimeError, match="do not end at epoch 10000"):
        analysis.collect_runs(config, latest_checkpoints=["NAMTM"])


def test_table_uses_final_metrics_even_when_earlier_epochs_are_better(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    for seed, final_epoch in enumerate((10000, 8500, 8000)):
        peak_epoch = 100 + seed
        metrics = {
            f"test_{metric}": [(100, 0.1), (101, 0.1), (102, 0.1), (final_epoch, 0.1)]
            for metric in analysis.METRICS
        }
        metrics["test_full_maze_accuracy"][seed] = (peak_epoch, 0.5 + 0.1 * seed)
        torch.save(
            {"metrics": metrics},
            tmp_path / f"runs/{seed}/checkpoint_epoch_{final_epoch}.pt",
        )
    _, _, per_seed = analysis.collect_runs(config, latest_checkpoints=["NAMTM"])
    assert [row["final_epoch"] for row in per_seed] == [10000, 8500, 8000]
    summary = analysis.aggregate(per_seed)[0]
    assert summary["mean_test_full_maze_accuracy"] == pytest.approx(0.1)
    assert summary["std_test_full_maze_accuracy"] == pytest.approx(0.0)


def test_latest_evaluation_includes_current_log_and_preserves_checkpoint_precision(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    log = tmp_path / "runs/0/wandb/run-20260921/files/output.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Test Epoch 10000 current: loss=0.2, accuracy=0.8, full_maze_accuracy=0.1000,\n"
        "Test Epoch 10005 current: loss=0.2, accuracy=0.8, full_maze_accuracy=0.7000,\n"
        "Test Epoch 10010 EMA:     loss=0.1, accuracy=0.9, full_maze_accuracy=0.9000,\n"
    )
    runs, _, rows = analysis.collect_runs(config, latest_checkpoints=["NAMTM"])
    assert runs["NAMTM"][0]["curves"]["loss"][10000] == 0.1
    assert rows[0]["final_epoch"] == 10005
    assert rows[0]["test_full_maze_accuracy"] == 0.7
    assert rows[0]["continuation_logs"].endswith("output.log")


def test_json_loader_reads_only_current_test_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        **{f"test_{metric}": [[1, 0.123456789], [5, 0.2]] for metric in analysis.METRICS},
        "test_ema_full_maze_accuracy": [[1, 0.99]],
        "train_full_maze_accuracy": [[1, 1.0]],
    }))
    curves = analysis.load_json_curves(path)
    assert curves["full_maze_accuracy"] == {1: 0.123456789, 5: 0.2}


def test_complete_metrics_do_not_require_cluster_logs(tmp_path, monkeypatch):
    config = make_runs(tmp_path, monkeypatch)
    for seed in range(3):
        path = tmp_path / "runs" / str(seed) / "metrics.json"
        path.write_text(json.dumps({f"test_{metric}": [[9000, 0.2 + 0.1 * seed]]
                                    for metric in analysis.METRICS}))
    _, _, rows = analysis.collect_runs(config)
    assert [row["final_epoch"] for row in rows] == [9000, 9000, 9000]
    assert all(row["submitit_log"] == "" for row in rows)
