"""Re-evaluate ImageNet100 checkpoints and export top-1/top-5 accuracy CSVs."""

import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from analysis.common import ROOT, CONFIGS, load_config
from tasks import construct_task


MODEL_ORDER = [
    "Transformer",
    "CMM",
    "CTM",
    "LSTM",
    "FSRNN",
    "RNN",
    "DNC",
    "LSAM",
    "NAMTM",
    "GDNStateTracking",
    "RMC",
    "RMT",
    "LSTM_CMM",
    "CTM_Matched",
    "CTMTransformer",
]

script_dir = os.path.dirname(os.path.abspath(__file__))


def find_final_checkpoint(run_dir):
    checkpoint_path = os.path.join(run_dir, "checkpoint_epoch_25.pt")
    return checkpoint_path if os.path.isfile(checkpoint_path) else None


def count_topk_correct(logits, targets, k):
    topk_predictions = logits.topk(k, dim=1).indices  # (B, num_classes) -> (B, k)
    top1_correct = (topk_predictions[:, 0] == targets).sum().item()  # (B), (B) -> ()
    topk_correct = (
        (topk_predictions == targets.unsqueeze(1)).any(dim=1).sum().item()
    )  # (B, k), (B) -> ()
    return top1_correct, topk_correct


def evaluate_checkpoint(checkpoint_path, device, num_workers, k):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    OmegaConf.set_struct(cfg, False)
    cfg.device = device.index if device.type == "cuda" else -1
    cfg.training.compile = False
    cfg.training.num_workers = num_workers
    cfg.model.backbone.pretrained = False
    OmegaConf.set_struct(cfg, True)

    task = construct_task(cfg, task_type=checkpoint["task_type"])
    state_dict = {
        name.removeprefix("_orig_mod.").replace("start_trace", "start_stm"): value
        for name, value in checkpoint["model_state_dict"].items()
    }
    task.model.load_state_dict(state_dict)
    task.model.to(device)
    task.model.eval()
    task.device = device

    total = 0
    correct_top1 = 0
    correct_topk = 0
    for inputs, targets in tqdm(
        task.test_dataloader,
        leave=False,
        desc=os.path.basename(os.path.dirname(checkpoint_path)),
    ):
        inputs = inputs.to(device)  # (B, C, H, W)
        targets = targets.to(device)  # (B)
        initial_states = task.model.get_initial_states(batch_size=inputs.size(0))
        with torch.inference_mode():
            logits, _ = task.model(
                inputs, states=initial_states
            )  # (B, num_classes, iterations)
        final_logits = logits[
            :, :, -1
        ]  # (B, num_classes, iterations) -> (B, num_classes)
        batch_top1, batch_topk = count_topk_correct(final_logits, targets, k)
        correct_top1 += batch_top1
        correct_topk += batch_topk
        total += targets.numel()

    return correct_top1 / total, correct_topk / total, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=None,
        help="Only re-evaluate these models, retaining cached rows for the others",
    )
    parser.add_argument("--config", type=Path, default=CONFIGS / "imagenet.yaml")
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/analysis/imagenet"
    )
    args = parser.parse_args()

    experiment_config = load_config(args.config, args.runs_root)

    device = torch.device(
        f"cuda:{args.device}"
        if args.device >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    per_seed_path = os.path.join(out_dir, f"top1_top{args.top_k}_per_seed.csv")

    rows = []
    selected_models = args.models or MODEL_ORDER
    if args.models and os.path.isfile(per_seed_path):
        with open(per_seed_path, newline="") as f:
            cached_rows = list(csv.DictReader(f))
        expected_fields = {
            "model",
            "seed",
            "num_examples",
            "top1_accuracy",
            f"top{args.top_k}_accuracy",
        }
        if cached_rows and set(cached_rows[0]) == expected_fields:
            rows.extend(
                row
                for row in cached_rows
                if row["model"] in MODEL_ORDER and row["model"] not in selected_models
            )

    for model in selected_models:
        base_dir = experiment_config["models"][model]
        for seed in (0, 1, 2):
            run_dir = os.path.join(base_dir, str(seed))
            checkpoint_path = find_final_checkpoint(run_dir)
            if checkpoint_path is None:
                print(
                    f"WARN: missing epoch-25 checkpoint for {model} seed {seed}: {run_dir}"
                )
                continue
            print(f"Evaluating {model} seed {seed}: {checkpoint_path}")
            top1_accuracy, topk_accuracy, num_examples = evaluate_checkpoint(
                checkpoint_path,
                device,
                args.num_workers,
                args.top_k,
            )
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "num_examples": num_examples,
                    "top1_accuracy": f"{top1_accuracy:.6f}",
                    f"top{args.top_k}_accuracy": f"{topk_accuracy:.6f}",
                }
            )
            print(f"  top-1={top1_accuracy:.4f}, top-{args.top_k}={topk_accuracy:.4f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No completed checkpoints found")

    model_rank = {model: rank for rank, model in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda row: (model_rank[row["model"]], int(row["seed"])))
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        top1_values = np.array([float(row["top1_accuracy"]) for row in model_rows])
        topk_values = np.array(
            [float(row[f"top{args.top_k}_accuracy"]) for row in model_rows]
        )
        summary_rows.append(
            {
                "model": model,
                "n_seeds": len(model_rows),
                "mean_top1_accuracy": f"{top1_values.mean():.6f}",
                "std_top1_accuracy": f"{top1_values.std(ddof=1) if len(top1_values) > 1 else 0.0:.6f}",
                f"mean_top{args.top_k}_accuracy": f"{topk_values.mean():.6f}",
                f"std_top{args.top_k}_accuracy": f"{topk_values.std(ddof=1) if len(topk_values) > 1 else 0.0:.6f}",
            }
        )

    summary_path = os.path.join(out_dir, f"top1_top{args.top_k}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {per_seed_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
