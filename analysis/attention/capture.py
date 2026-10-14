"""Capture memory attention without changing the model forward pass."""

from contextlib import contextmanager

import torch


def attention_shares(attention, num_ltm):
    """(B,H,S,S) -> (B,H,2,3), queries LTM/STM, keys sink/LTM/STM."""
    if attention.ndim != 4 or attention.shape[-2] != attention.shape[-1]:
        raise ValueError("Expected square attention matrices (B,H,S,S)")
    if not 0 < num_ltm < attention.shape[-1] - 1:
        raise ValueError("Both LTM and STM tokens must be present")
    destinations = torch.stack(
        (
            attention[..., 0],
            attention[..., 1 : 1 + num_ltm].sum(-1),
            attention[..., 1 + num_ltm :].sum(-1),
        ),
        dim=-1,
    )
    return torch.stack(
        (
            destinations[:, :, 1 : 1 + num_ltm].mean(2),
            destinations[:, :, 1 + num_ltm :].mean(2),
        ),
        dim=2,
    )


@contextmanager
def capture_attention(cell):
    """Capture only batch sums, without retaining full matrices or changing outputs."""
    captured = [[] for _ in cell.memory_updater.blocks]
    handles = []
    for layer, block in enumerate(cell.memory_updater.blocks):

        def hook(module, args, output, layer=layer):
            shares = attention_shares(
                module._last_attn, cell.num_long_term_memory_tokens
            )
            captured[layer].append(shares.double().sum(0))

        handles.append(block.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
