import pytest
import torch
from models.base import RecurrentModel
from models.cells.cmm import ContinuousMemoryMachine


def test_visual_attention_budgets_match_explicit_token_sums():
    from analysis.attention.capture import attention_shares

    torch.manual_seed(42)
    attention = torch.randn(2, 3, 8, 8).softmax(-1)
    actual = attention_shares(attention, num_ltm=4)
    expected = torch.empty(2, 3, 2, 3)
    for batch in range(2):
        for head in range(3):
            for q, queries in enumerate((range(1, 5), range(5, 8))):
                for d, keys in enumerate((range(1), range(1, 5), range(5, 8))):
                    expected[batch, head, q, d] = sum(
                        attention[batch, head, i, j] for i in queries for j in keys
                    ) / len(queries)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, 3, 2))


def test_visual_attention_capture_preserves_logits_and_batch_weighting():
    from analysis.attention.capture import attention_shares, capture_attention

    cell = ContinuousMemoryMachine(
        8,
        8,
        3,
        0,
        num_long_term_memory_tokens=2,
        memory_transformer_layers=2,
        num_heads=2,
    )
    model = RecurrentModel(cell).eval()
    inputs = torch.randn(4, 3, 8)
    with torch.inference_mode():
        expected, _ = model(inputs, model.get_initial_state(3))
        with capture_attention(cell) as captured:
            actual, _ = model(inputs, model.get_initial_state(3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(captured) == 2 and all(len(layer) == 4 for layer in captured)
        for block, layer in zip(cell.memory_updater.blocks, captured):
            torch.testing.assert_close(
                layer[-1], attention_shares(block._last_attn, 2).double().sum(0)
            )
            torch.testing.assert_close(
                layer[-1].sum(-1), torch.full((2, 2), 3.0, dtype=torch.float64)
            )
            assert not block._forward_hooks
        with pytest.raises(RuntimeError), capture_attention(cell):
            raise RuntimeError("failed capture")
        assert all(not block._forward_hooks for block in cell.memory_updater.blocks)
