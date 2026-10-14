"""Render reproducible ImageNet100 attention trajectories and a candidate gallery."""

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from PIL import Image
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def compute_com(attn_weights, grid_h, grid_w):
    """Return (y, x) in patch coordinates for nonnegative patch attention."""
    attn_2d = np.asarray(attn_weights).reshape(grid_h, grid_w)
    yy, xx = np.mgrid[:grid_h, :grid_w]
    total = attn_2d.sum() + 1e-12
    return (attn_2d * yy).sum() / total, (attn_2d * xx).sum() / total


def trajectory_pixels(attention, image_shape):
    """Map square-grid patch centers to imshow pixel centers, retaining every tick."""
    grid = math.isqrt(attention.shape[-1])
    if grid * grid != attention.shape[-1]:
        raise ValueError("Expected a square patch grid without a class token")
    coms = np.array([compute_com(weights, grid, grid) for weights in attention])
    return (coms + 0.5) * (np.array(image_shape[:2]) / grid) - 0.5


def prepare_imagenet_image(inputs):
    """Invert ImageNet normalization to display the exact evaluation crop in RGB."""
    image = inputs[0].detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)


def draw_trajectory(ax, image, attention):
    """Use the original paper bird's Spectral colors, curved arrows and dark halos."""
    positions = trajectory_pixels(attention, image.shape)
    ticks = len(positions)
    colors = plt.colormaps["Spectral"](np.linspace(0, 1, ticks))
    ax.imshow(image, interpolation="lanczos")
    shadow = [
        patheffects.Stroke(linewidth=4, foreground="black", alpha=0.6),
        patheffects.Normal(),
    ]
    for tick in range(ticks - 1):
        arrow = ax.annotate(
            "",
            xy=positions[tick + 1, ::-1],
            xytext=positions[tick, ::-1],
            arrowprops=dict(
                arrowstyle="-|>",
                color=colors[tick],
                lw=1.8,
                mutation_scale=15,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=4 + tick,
        )
        arrow.arrow_patch.set_path_effects(shadow)
    marker_shadow = [
        patheffects.Stroke(linewidth=3, foreground="black", alpha=0.7),
        patheffects.Normal(),
    ]
    for tick, marker, size in [(0, "o", 10), (-1, "s", 11)]:
        ax.plot(
            positions[tick, 1],
            positions[tick, 0],
            marker,
            color=colors[tick],
            markersize=size,
            zorder=ticks + 6,
            path_effects=marker_shadow,
        )
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.axis("off")


def save_figure(fig, path, dpi):
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        path.with_name(path.name + "_thumb").with_suffix(".jpg"),
        dpi=60,
        bbox_inches="tight",
    )
    plt.close(fig)


def render_example(image, attention, out_dir, stem, heads, dpi):
    """Export the mean, head overview, and requested individual heads."""
    variants = [("Mean across heads", stem, attention.mean(axis=1))]
    variants += [
        (f"Head {head}", f"{stem}_head{head}", attention[:, head]) for head in heads
    ]
    cards = []
    for title, name, weights in variants:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_trajectory(ax, image, weights)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_figure(fig, out_dir / name, dpi)
        cards.append({"title": title, "file": name})

    n_heads = attention.shape[1]
    cols = min(2, n_heads)
    rows = math.ceil(n_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for head, ax in enumerate(axes.flat):
        if head < n_heads:
            draw_trajectory(ax, image, attention[:, head])
        else:
            ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    overview = stem.replace("attention_com_", "attention_com_heads_", 1)
    save_figure(fig, out_dir / overview, dpi)
    return cards


def capture_attention(model, inputs):
    """Capture actual cross-attention weights without changing model parameters."""
    module = model.rnn.cross_attn
    previous = module.return_weights
    per_tick = []
    module.return_weights = True
    handle = module.register_forward_hook(
        lambda _module, _inputs, outputs: per_tick.append(outputs[1].detach().cpu())
    )
    try:
        with torch.inference_mode():
            logits, _ = model(inputs, states=model.get_initial_states(inputs.shape[0]))
    finally:
        handle.remove()
        module.return_weights = previous
    attention = torch.stack(per_tick)[:, 0].numpy()  # (ticks, heads, patches)
    probabilities = logits[0].softmax(dim=0).detach().cpu().numpy()  # (classes, ticks)
    return attention, probabilities


def sample_indices(dataset_size, seed, count, indices=None):
    """Keep legacy seed-to-image mapping and avoid duplicate images in a batch."""
    if indices is not None:
        if any(index < 0 or index >= dataset_size for index in indices):
            raise ValueError(f"Image indices must be in [0, {dataset_size})")
        return [(None, index) for index in dict.fromkeys(indices)]
    if count < 1 or count > dataset_size:
        raise ValueError(f"--num-images must be between 1 and {dataset_size}")
    selected, seen = [], set()
    while len(selected) < count:
        index = int(np.random.RandomState(seed).randint(dataset_size))
        if index not in seen:
            selected.append((seed, index))
            seen.add(index)
        seed += 1
    return selected


def write_gallery(out_dir, records):
    sections = []
    for record in records:
        name = html.escape(record["true_name"])
        prediction = html.escape(record["predicted_name"])
        status = "correct" if record["correct"] else "incorrect"
        stem = record["stem"]
        cards = [
            f'<figure><a href="{stem}_input.png"><img loading="lazy" '
            f'src="{stem}_input.png"></a><figcaption>Input crop</figcaption></figure>'
        ]
        for card in record["variants"]:
            path = card["file"]
            title = html.escape(card["title"])
            cards.append(
                f'<figure><a href="{path}.png"><img loading="lazy" src="{path}_thumb.jpg"></a>'
                f'<figcaption>{title} · <a href="{path}.pdf">PDF</a> · '
                f'<a href="{path}.png">PNG</a></figcaption></figure>'
            )
        sections.append(
            f'<section data-correct="{str(record["correct"]).lower()}">'
            f"<h2>{name} · index {record['dataset_index']} · seed {record['sample_seed']}</h2>"
            f"<p>Prediction: {prediction} ({record['confidence']:.1%}, {status}). "
            f'<a href="{stem}.json">Metadata</a> · <a href="{stem}.npz">Attention arrays</a></p>'
            f'<div class="images">{"".join(cards)}</div></section>'
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>CMM attention candidates</title>
<style>
body{font:16px system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
.controls{position:sticky;top:0;background:#fafafa;padding:12px 0;z-index:1}
input[type=search]{font:inherit;padding:8px;width:320px}
section{border-top:1px solid #ccc;padding:16px 0}h2{font-size:20px;margin:0}
.images{display:flex;gap:8px;overflow-x:auto}figure{margin:0;flex:1;min-width:200px}
img{width:100%;display:block}figcaption{padding:8px 0}a{color:#2453a1}
</style><h1>CMM attention candidates</h1>
<p>All ticks are shown: red → yellow → green → blue. Circle = start; square = end.
Arrows trace attention centers of mass; compare individual heads before choosing an example.</p>
<div class="controls"><input id="search" type="search" placeholder="Filter class, index or seed">
<label><input id="correct" type="checkbox"> Correct predictions only</label></div>
"""
        + "\n".join(sections)
        + """
<script>
function filter(){
const query=document.getElementById('search').value.toLowerCase();
const correct=document.getElementById('correct').checked;
document.querySelectorAll('section').forEach(s=>{
s.hidden=!s.textContent.toLowerCase().includes(query)||(correct&&s.dataset.correct!=='true');
});
}
document.getElementById('search').addEventListener('input',filter);
document.getElementById('correct').addEventListener('change',filter);
</script></html>
"""
    )
    (out_dir / "index.html").write_text(document)
    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


def load_model(checkpoint_path, device_id):
    from omegaconf import OmegaConf
    from tasks import construct_task

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["task_type"] != "ImageNet100":
        raise ValueError("This renderer expects an ImageNet100 checkpoint")
    cfg = OmegaConf.create(checkpoint["cfg"])
    cfg.device = device_id if device_id != -1 and torch.cuda.is_available() else -1
    cfg.training.compile = False
    cfg.training.num_workers = 0
    cfg.training.setdefault("use_scheduler", False)
    cfg.logging.wandb.enabled = False
    cfg.model.backbone.pretrained = False
    torch.manual_seed(cfg.seed)
    if cfg.device == -1:
        torch.set_num_threads(4)
    task = construct_task(cfg, task_type=checkpoint["task_type"])
    state_dict = {
        key.replace("_orig_mod.", "").replace("start_trace", "start_stm"): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    task.model.load_state_dict(state_dict, strict=True)
    task.model.eval()
    if not hasattr(task.model.rnn, "cross_attn"):
        raise ValueError("The model must have cross-attention over image patches")
    print(f"Loaded epoch {checkpoint['epoch']} on {task.device}", flush=True)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint .pt file")
    parser.add_argument(
        "--device", type=int, default=0, help="GPU index, or -1 for CPU"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="First image-selection seed"
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of distinct test crops"
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", help="Exact test indices instead of seeds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "outputs/analysis/image_classification",
    )
    heads = parser.add_mutually_exclusive_group()
    heads.add_argument(
        "--head", type=int, help="Export this individual head (zero based)"
    )
    heads.add_argument(
        "--all-heads", action="store_true", help="Export every head separately"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution; PDFs retain vector arrows"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the local Hugging Face dataset cache",
    )
    args = parser.parse_args()
    if args.num_images < 1 or args.dpi < 1 or args.seed < 0 or args.device < -1:
        parser.error("Image count and dpi must be positive; seed >= 0 and device >= -1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    task = load_model(args.checkpoint, args.device)
    n_heads = task.model.rnn.cross_attn.num_heads
    if args.head is not None and not 0 <= args.head < n_heads:
        parser.error(f"--head must be between 0 and {n_heads - 1}")
    selected_heads = (
        list(range(n_heads))
        if args.all_heads
        else ([args.head] if args.head is not None else [])
    )
    from analysis.imagenet.labels import _get_class_names

    class_names = _get_class_names(task)
    selections = sample_indices(
        len(task.test_dataset), args.seed, args.num_images, args.indices
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, (seed, index) in enumerate(selections, 1):
        input_tensor, label = task.test_dataset[index]
        inputs = input_tensor.unsqueeze(0).to(task.device)
        attention, probabilities = capture_attention(task.model, inputs)
        image = prepare_imagenet_image(inputs)
        pred = int(probabilities[:, -1].argmax())
        label = int(label)
        true_name = str(class_names[label]) if class_names else str(label)
        pred_name = str(class_names[pred]) if class_names else str(pred)
        stem = (
            f"attention_com_seed{seed}"
            if seed is not None
            else f"attention_com_index{index}"
        )
        Image.fromarray(np.round(image * 255).astype(np.uint8)).save(
            args.output_dir / f"{stem}_input.png"
        )
        variants = render_example(
            image, attention, args.output_dir, stem, selected_heads, args.dpi
        )
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            attention=attention,
            probabilities=probabilities,
            com_yx_pixels=np.stack(
                [
                    trajectory_pixels(attention[:, head], image.shape)
                    for head in range(n_heads)
                ],
                axis=1,
            ),
        )
        record = dict(
            stem=stem,
            checkpoint=str(args.checkpoint.resolve()),
            split="test",
            dataset_index=index,
            sample_seed=seed,
            true_label=label,
            true_name=true_name,
            predicted_label=pred,
            predicted_name=pred_name,
            confidence=float(probabilities[pred, -1]),
            correct=pred == label,
            ticks=int(attention.shape[0]),
            heads=n_heads,
            attention_shape=list(attention.shape),
            variants=variants,
            style="Spectral; curved arrows; translucent black outlines; circle start, square end",
        )
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        records.append(record)
        write_gallery(args.output_dir, records)
        print(
            f"[{number}/{len(selections)}] {stem}: {true_name} -> {pred_name} "
            f"({record['confidence']:.1%})",
            flush=True,
        )
    print(f"Gallery: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
