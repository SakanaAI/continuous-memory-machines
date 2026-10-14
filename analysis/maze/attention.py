"""Paper figures for the Maze task:
1. Maze with final-tick rainbow path + per-head cross-attention COM arrows

Usage:
    python -m analysis.maze.attention <checkpoint> --seed 110 [--device -1]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
import torch

mpl.use("Agg")
sns.set_style("whitegrid")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.checkpoints import load_cmm_task
from analysis.maze.drawing import _draw_predicted_path, _prepare_maze_image


# ── helpers ──────────────────────────────────────────────────────────────────


def compute_com(attn_weights, grid_h, grid_w):
    """Center-of-mass in patch coordinates from attention weights (N,)."""
    attn_2d = attn_weights.reshape(grid_h, grid_w)
    yy, xx = np.mgrid[:grid_h, :grid_w]
    total = attn_2d.sum() + 1e-12
    cy = (attn_2d * yy).sum() / total
    cx = (attn_2d * xx).sum() / total
    return cy, cx


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: maze + rainbow path + per-head COM arrows ─────────────────────


def fig_maze_attention_com(
    maze_image,
    predicted_actions_last_tick,
    attn_stack,
    image_h,
    image_w,
    output_dir,
    seed,
):
    """Maze with final-tick rainbow path overlaid with per-head COM arrow trajectories.

    attn_stack: (T, H, N) — per-tick, per-head attention over N patches.
    """
    T, H, N = attn_stack.shape
    grid = int(N**0.5)
    patch_h = image_h / grid
    patch_w = image_w / grid

    rainbow = plt.get_cmap("gist_rainbow")
    maze_with_path = _draw_predicted_path(
        maze_image, predicted_actions_last_tick, rainbow
    )

    cmap = plt.cm.Spectral
    colors = [cmap(t / max(T - 1, 1)) for t in range(T)]
    shadow = [
        patheffects.Stroke(linewidth=3.5, foreground="white", alpha=0.9),
        patheffects.Normal(),
    ]

    def _draw_com_arrows(ax, coms_px, colors, T, lw=1.6, mutation_scale=12):
        """Draw COM arrows on an axis, skipping the first arrow and tiny movements."""
        min_dist = 0.3  # skip arrows shorter than this in pixel coords
        for t in range(1, T - 1):  # start from 1 to skip first arrow
            dx = coms_px[t + 1, 0] - coms_px[t, 0]
            dy = coms_px[t + 1, 1] - coms_px[t, 1]
            if (dx**2 + dy**2) ** 0.5 < min_dist:
                continue
            arr = ax.annotate(
                "",
                xy=(coms_px[t + 1, 0], coms_px[t + 1, 1]),
                xytext=(coms_px[t, 0], coms_px[t, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors[t],
                    lw=lw,
                    mutation_scale=mutation_scale,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc3,rad=0.08",
                ),
                zorder=4 + t,
            )
            arr.arrow_patch.set_path_effects(shadow)
        # End marker only (no start marker)
        end_shadow = [
            patheffects.Stroke(linewidth=3, foreground="white", alpha=0.9),
            patheffects.Normal(),
        ]
        ax.plot(
            coms_px[-1, 0],
            coms_px[-1, 1],
            "s",
            color=colors[-1],
            markersize=7,
            zorder=T + 6,
            path_effects=end_shadow,
        )

    # Individual per-head images
    for hi in range(H):
        attn_h = attn_stack[:, hi, :]  # (T, N)
        coms_h = np.array([compute_com(attn_h[t], grid, grid) for t in range(T)])
        coms_h_px = np.column_stack(
            [
                (coms_h[:, 1] + 0.5) * patch_w,
                (coms_h[:, 0] + 0.5) * patch_h,
            ]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(maze_with_path, interpolation="nearest")
        _draw_com_arrows(ax, coms_h_px, colors, T, lw=2.0, mutation_scale=15)
        ax.set_xlim(-0.5, image_w - 0.5)
        ax.set_ylim(image_h - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        _savefig(fig, f"{output_dir}/maze_attention_com_seed{seed}_head{hi}.png")
        _savefig(fig, f"{output_dir}/maze_attention_com_seed{seed}_head{hi}.pdf")

    # Overlay pairs of heads on the same image with different cmaps
    head_pairs = [(0, 7)] if H >= 8 else []
    cmap_a, cmap_b = plt.cm.winter, plt.cm.autumn
    for ha, hb in head_pairs:
        colors_a = [cmap_a(t / max(T - 1, 1)) for t in range(T)]
        colors_b = [cmap_b(t / max(T - 1, 1)) for t in range(T)]

        coms_a = np.array(
            [compute_com(attn_stack[:, ha, :][t], grid, grid) for t in range(T)]
        )
        coms_a_px = np.column_stack(
            [(coms_a[:, 1] + 0.5) * patch_w, (coms_a[:, 0] + 0.5) * patch_h]
        )
        coms_b = np.array(
            [compute_com(attn_stack[:, hb, :][t], grid, grid) for t in range(T)]
        )
        coms_b_px = np.column_stack(
            [(coms_b[:, 1] + 0.5) * patch_w, (coms_b[:, 0] + 0.5) * patch_h]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(maze_with_path, interpolation="nearest")
        _draw_com_arrows(ax, coms_a_px, colors_a, T, lw=2.0, mutation_scale=15)
        _draw_com_arrows(ax, coms_b_px, colors_b, T, lw=2.0, mutation_scale=15)
        ax.set_xlim(-0.5, image_w - 0.5)
        ax.set_ylim(image_h - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        _savefig(
            fig, f"{output_dir}/maze_attention_com_seed{seed}_head{ha}_head{hb}.png"
        )
        _savefig(
            fig, f"{output_dir}/maze_attention_com_seed{seed}_head{ha}_head{hb}.pdf"
        )

    # Also save avg-head version (single image)
    attn_avg = attn_stack.mean(axis=1)  # (T, N)
    coms_avg = np.array([compute_com(attn_avg[t], grid, grid) for t in range(T)])
    coms_avg_px = np.column_stack(
        [
            (coms_avg[:, 1] + 0.5) * patch_w,
            (coms_avg[:, 0] + 0.5) * patch_h,
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze_with_path, interpolation="nearest")
    _draw_com_arrows(ax, coms_avg_px, colors, T, lw=2.0, mutation_scale=15)

    ax.set_xlim(-0.5, image_w - 0.5)
    ax.set_ylim(image_h - 0.5, -0.5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _savefig(fig, f"{output_dir}/maze_attention_com_avg_seed{seed}.png")
    _savefig(fig, f"{output_dir}/maze_attention_com_avg_seed{seed}.pdf")


# ── Figure 2: LTM / STM attention budget ────────────────────────────────────


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.device >= 0 and not torch.cuda.is_available():
        parser.error("Requested CUDA device unavailable; use --device -1 for CPU")
    task, cfg, task_type, epoch = load_cmm_task(
        args.checkpoint, args.device, num_workers=0
    )
    if task_type != "Maze":
        parser.error("Expected a Maze checkpoint")
    device = task.device

    # Pick test maze
    rng = np.random.RandomState(args.seed)
    idx = rng.randint(len(task.test_dataset))
    img, target = task.test_dataset[idx]
    inputs = img.unsqueeze(0).to(device)
    non_wait = (target != 4).sum().item()
    print(f"Seed {args.seed} -> test maze index {idx}, non_wait_steps={non_wait}")

    # Hook cross-attention weights
    attn_module = task.model.rnn.cross_attn
    attn_per_tick = []
    attn_module.return_weights = True
    handle = attn_module.register_forward_hook(
        lambda _m, _i, _o: attn_per_tick.append(_o[1].clone())
    )

    print("Running forward pass...")
    try:
        with torch.inference_mode():
            initial_states = task.model.get_initial_states(batch_size=inputs.size(0))
            logits, _ = task.model(inputs, states=initial_states)
    finally:
        handle.remove()
        attn_module.return_weights = False

    # Check correctness at last tick
    route_length = task.route_length
    T = logits.shape[-1]
    predictions = logits.reshape(1, route_length, 5, T)
    pred_last = predictions[:, :, :, -1].argmax(dim=2)[0].cpu()
    solved = (pred_last == target).all().item()
    print(f"Solved at last tick: {solved}")

    # Cross-attention: (T, B, H, N) -> (T, H, N) for batch 0
    attn_stack = torch.stack(attn_per_tick, dim=0)[:, 0].cpu().numpy()  # (T, H, N)

    # Predicted actions at last tick for the rainbow path
    predicted_actions_last = (
        predictions.argmax(dim=2)[0, :, -1].cpu().numpy()
    )  # (route_length,)

    maze_image = _prepare_maze_image(inputs)
    image_h, image_w = maze_image.shape[:2]

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_parent = Path(args.checkpoint).resolve().parent
        run_id = (
            ckpt_parent.parent.name if ckpt_parent.name.isdigit() else ckpt_parent.name
        )
        output_dir = str(_REPO_ROOT / "outputs" / "analysis" / "maze_paper" / run_id)

    print(f"\nGenerating figures -> {output_dir}")

    # Figure 1: maze + rainbow path + COM arrows
    fig_maze_attention_com(
        maze_image,
        predicted_actions_last,
        attn_stack,
        image_h,
        image_w,
        output_dir,
        args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
