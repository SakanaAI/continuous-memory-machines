import pytest
import torch
from omegaconf import OmegaConf

from analysis.imagenet.evaluate import count_topk_correct
from models import construct_rnn
from models.base import CrossAttentionRecurrentModel
from models.cells.gated_delta_net import CrossAttentionGatedDeltaNetCell
from models.cells.lsam import CrossAttentionLSAMCell
from models.cells.lstm_cmm import CrossAttentionLSTMCMMCell
from models.cells.rmc import CrossAttentionRelationalMemoryCore
from tasks.image_classification.base import ImageClassificationModel
from tasks.maze.maze import MazeModel, maze_loss


@pytest.mark.parametrize(
    ("cell", "query_dim"),
    [
        (CrossAttentionLSAMCell(d_input=16, d_model=8, num_heads=4), 8),
        (CrossAttentionGatedDeltaNetCell(d_input=16, d_model=8, num_heads=4), 8),
        (CrossAttentionRelationalMemoryCore(d_input=16, d_model=8, mem_slots=4, num_heads=4), 16),
        (CrossAttentionLSTMCMMCell(d_input=16, d_model=8, num_long_term_memory_tokens=4, num_heads=4), 16),
    ],
)
def test_structured_state_cross_attention_cells(cell, query_dim):
    model = CrossAttentionRecurrentModel(cell, num_heads=4)
    inputs = torch.randn(3, 2, 5, 16)  # (iterations, B, N, d_token)
    initial_states = model.get_initial_state(batch_size=2)

    outputs, final_states = model(inputs, initial_states)

    assert outputs.shape == (3, 2, 8)  # (iterations, B, d_model)
    assert final_states[-1].shape == (2, query_dim)
    assert model.cross_attn.q_proj.in_features == query_dim


def test_patch_mode_causal_transformer_caches_internal_ticks():
    cfg = OmegaConf.create(
        {
            "model": "Transformer",
            "d_input": 16,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_seq_len": 4,
            "learn_pos_emb": False,
            "causal": True,
            "backbone": {"mode": "patches", "num_heads": 4},
        }
    )
    model = construct_rnn(cfg).eval()
    inputs = torch.randn(3, 2, 5, 16)  # (ticks, B, image patches, d_token)

    outputs, final_state = model(inputs, model.get_initial_state(batch_size=2))

    assert outputs.shape == (3, 2, 8)
    assert len(final_state) == 5  # K/V cache per layer, followed by query state
    for cache in final_state[:-1]:
        assert cache.shape == (2, 2, 3, 4)
    torch.testing.assert_close(final_state[-1], outputs[-1])

    changed_future = inputs.clone()
    changed_future[-1] = torch.randn_like(changed_future[-1])
    changed_outputs, _ = model(
        changed_future, model.get_initial_state(batch_size=2)
    )
    torch.testing.assert_close(changed_outputs[:-1], outputs[:-1])


def test_patch_mode_transformer_requires_causal_attention():
    cfg = OmegaConf.create(
        {
            "model": "Transformer",
            "d_input": 16,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_seq_len": 4,
            "learn_pos_emb": False,
            "causal": False,
            "backbone": {"mode": "patches", "num_heads": 4},
        }
    )

    with pytest.raises(ValueError, match="requires model.causal=true"):
        construct_rnn(cfg)


def _patch_transformer_cfg():
    return OmegaConf.create(
        {
            "model": "Transformer",
            "d_input": 16,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_seq_len": 4,
            "learn_pos_emb": False,
            "causal": True,
            "backbone": {
                "type": "resnet",
                "arch": "resnet18",
                "mode": "patches",
                "pretrained": False,
                "freeze": False,
                "d_model": 16,
                "num_blocks": 1,
                "num_heads": 4,
                "pool": True,
            },
        }
    )


def test_image_classification_model_supports_tick_transformer():
    model = ImageClassificationModel(
        _patch_transformer_cfg(), OmegaConf.create({"iterations": 3, "out_dims": 10})
    )
    inputs = torch.randn(2, 3, 8, 8)

    logits, final_state = model(inputs, model.get_initial_states(batch_size=2))

    assert logits.shape == (2, 10, 3)
    assert final_state[0].size(2) == 3


def test_maze_model_supports_tick_transformer_and_loss():
    model = MazeModel(
        _patch_transformer_cfg(),
        OmegaConf.create({"iterations": 3, "out_dims": 20}),
    )
    inputs = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 5, (2, 4))

    logits, _ = model(inputs, model.get_initial_states(batch_size=2))
    predictions = logits.reshape(2, 4, 5, 3)
    loss, _, _ = maze_loss(predictions, targets)
    loss.backward()

    assert logits.shape == (2, 20, 3)
    assert torch.isfinite(loss)


def test_count_topk_correct():
    logits = torch.tensor([[9, 8, 7, 6, 5, 4], [9, 8, 7, 6, 5, 4], [9, 8, 7, 6, 5, 4]])  # (B, num_classes)
    targets = torch.tensor([0, 2, 5])  # (B)

    top1_correct, top5_correct = count_topk_correct(logits, targets, 5)

    assert top1_correct == 1
    assert top5_correct == 2
