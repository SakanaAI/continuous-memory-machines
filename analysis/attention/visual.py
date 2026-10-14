"""Export intact CMM attention over the complete Maze and ImageNet-100 test sets."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from analysis.attention.capture import capture_attention
from analysis.checkpoints import load_cmm_task, seed_all
from analysis.common import CONFIGS, ROOT, load_config, write_csv


@torch.inference_mode()
def evaluate(checkpoint, setting, seed, args):
    task, cfg, task_type, epoch = load_cmm_task(
        checkpoint, args.device, args.num_workers
    )
    if task_type != setting or int(cfg.seed) != seed:
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
    dataset = task.test_dataloader.dataset
    if args.max_examples:
        dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))
    batch_size = args.batch_size or int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device >= 0,
    )
    seed_all(args.eval_seed)
    cell = task.model.rnn.recurrent_module
    totals, count = None, 0
    digest = hashlib.sha256()
    for inputs, targets in loader:
        digest.update(inputs.numpy().tobytes())
        digest.update(targets.numpy().tobytes())
        inputs = inputs.to(task.device)
        with capture_attention(cell) as captured:
            task.model(inputs, task.model.get_initial_states(inputs.size(0)))
        batch = torch.stack([torch.stack(layer) for layer in captured]).cpu().numpy()
        if batch.shape[1] != task.model.iterations:
            raise RuntimeError("Missing attention ticks")
        totals = batch if totals is None else totals + batch
        count += inputs.size(0)
        if (
            count == inputs.size(0)
            or count == len(dataset)
            or count % (10 * batch_size) == 0
        ):
            print(f"{setting} seed={seed}: {count}/{len(dataset)}", flush=True)
    if not count or count != len(dataset):
        raise RuntimeError("Incomplete attention evaluation")
    mean = totals / count
    if not np.isfinite(mean).all() or not np.allclose(mean.sum(-1), 1, atol=2e-6):
        raise RuntimeError("Attention mass is not conserved")
    rows = [
        {
            "task": setting,
            "seed": seed,
            "condition": "intact",
            "layer": layer,
            "tick": tick + 1,
            "head": head,
            "query": ("LTM", "STM")[query],
            "destination": ("Sink", "LTM", "STM")[destination],
            "mean_attention": float(mean[layer, tick, head, query, destination]),
            "num_examples": count,
        }
        for layer, tick, head, query, destination in np.ndindex(mean.shape)
    ]
    info = {
        "task": setting,
        "seed": seed,
        "epoch": epoch,
        "num_examples": count,
        "checkpoint": str(checkpoint),
        "eval_seed": args.eval_seed,
        "batch_size": batch_size,
        "test_data_sha256": digest.hexdigest(),
        "num_ltm": cell.num_long_term_memory_tokens,
        "num_stm": cell.memory_length,
        "num_layers": len(cell.memory_updater.blocks),
    }
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "attention.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--batch-size", type=int, default=0, help="0 uses the checkpoint batch size"
    )
    parser.add_argument("--eval-seed", type=int, default=20260917)
    parser.add_argument(
        "--max-examples", type=int, default=0, help="0 uses the entire test split"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("Maze", "ImageNet100"),
        default=["Maze", "ImageNet100"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/attention/visual"
    )
    args = parser.parse_args()
    if min(args.max_examples, args.batch_size, args.num_workers) < 0:
        parser.error("Example count, batch size, and worker count must be nonnegative")
    if args.device >= 0 and not torch.cuda.is_available():
        parser.error("Requested CUDA device unavailable; use --device -1 for CPU")
    config = load_config(args.config, args.runs_root)
    for setting in args.tasks:
        run = config[setting]
        for seed in args.seeds:
            checkpoint = (
                Path(run["run"]) / str(seed) / f"checkpoint_epoch_{run['epoch']}.pt"
            )
            rows, info = evaluate(checkpoint, setting, seed, args)
            write_csv(args.output_dir / f"{setting}_seed{seed}.csv", rows)
            (args.output_dir / f"{setting}_seed{seed}.json").write_text(
                json.dumps(info, indent=2) + "\n"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
