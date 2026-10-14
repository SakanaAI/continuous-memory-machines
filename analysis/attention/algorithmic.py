"""Export the single-example Copy and Sort memory attention used in the paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.attention.capture import capture_attention
from analysis.checkpoints import load_cmm_task, seed_all
from analysis.common import CONFIGS, ROOT, load_config, write_csv


def evaluate(checkpoint, task_type, device, sample_seed, sample_index):
    task, cfg, actual_task, epoch = load_cmm_task(checkpoint, device, num_workers=0)
    if actual_task != task_type:
        raise ValueError(f"Expected {task_type} checkpoint, got {actual_task}")
    seed_all(sample_seed)
    for index, (inputs, targets) in enumerate(task.test_dataloader):
        if index == sample_index:
            break
    else:
        raise IndexError(f"Test batch {sample_index} does not exist")
    inputs, targets = inputs[:1].to(task.device), targets[:1].to(task.device)
    cell = task.model.rnn.recurrent_module
    with torch.inference_mode(), capture_attention(cell) as captured:
        logits, _ = task.model(inputs, states=task.model.get_initial_states(1))
        loss = task._compute_loss(logits, targets)
        metrics = task._compute_metrics(logits, targets, loss)
    # The original algorithmic figure uses the first memory layer, averaged over heads.
    budget = torch.stack(captured[0]).mean(1).cpu().numpy()
    np.testing.assert_allclose(budget.sum(-1), 1, atol=2e-6)
    boundary = (
        int(inputs[0, :, -1].nonzero()[0, 0])
        if task_type == "Copy"
        else task.num_input_items
    )
    rows = [
        {
            "task": "Copy" if task_type == "Copy" else "Sort",
            "timestep": step,
            "phase": "input" if step < boundary else "output",
            **{
                f"{query}_to_{destination}": float(budget[step, q, d])
                for q, query in enumerate(("ltm", "stm"))
                for d, destination in enumerate(("sink", "ltm", "stm"))
            },
        }
        for step in range(len(budget))
    ]
    return rows, {
        "checkpoint": str(checkpoint),
        "task": actual_task,
        "epoch": epoch,
        "training_seed": int(cfg.seed),
        "sample_seed": sample_seed,
        "sample_index": sample_index,
        "attention_block": 0,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "attention.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Override the per-task training seed in the manifest",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--sample-index", type=int, default=0, help="Test batch; use its first example"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/attention"
    )
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.device >= 0 and not torch.cuda.is_available():
        parser.error("Requested CUDA device unavailable; use --device -1 for CPU")
    config = load_config(args.config, args.runs_root)
    rows, metadata = [], []
    for task in ("Copy", "PrioritySort"):
        run = config[task]
        training_seed = (
            args.training_seed if args.training_seed is not None else run["seed"]
        )
        checkpoint = (
            Path(run["run"])
            / str(training_seed)
            / f"checkpoint_epoch_{run['epoch']}.pt"
        )
        task_rows, info = evaluate(
            checkpoint, task, args.device, args.sample_seed, args.sample_index
        )
        rows.extend(task_rows)
        metadata.append(info)
    write_csv(args.output_dir / "algorithmic.csv", rows)
    (args.output_dir / "algorithmic.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved Copy/Sort attention to {args.output_dir}")


if __name__ == "__main__":
    main()
