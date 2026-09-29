import torch

from tinyinfer.layers.rotary_embedding import RotaryEmbedding


def reference_apply_rotary(x, positions, base=10000.0):
    """独立写一个小 reference，避免测试直接复用被测函数内部实现。"""
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freqs = torch.outer(positions.float(), inv_freq)
    cos = freqs.cos().unsqueeze(1).to(x.dtype)
    sin = freqs.sin().unsqueeze(1).to(x.dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat(
        (x1 * cos - x2 * sin, x2 * cos + x1 * sin),
        dim=-1,
    )


def test_rope_mixed_positions_match_reference():
    torch.manual_seed(0)

    rope = RotaryEmbedding(head_dim=8, rotary_dim=8, base=10000.0)
    q = torch.randn(6, 4, 8)
    k = torch.randn(6, 2, 8)

    # decode/prefill/decode 混排，positions 故意不是单调递增。
    positions = torch.tensor([12, 4, 5, 6, 7, 29], dtype=torch.long)

    q_out, k_out = rope(q, k, positions)
    q_ref = reference_apply_rotary(q, positions)
    k_ref = reference_apply_rotary(k, positions)

    torch.testing.assert_close(q_out, q_ref)
    torch.testing.assert_close(k_out, k_ref)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(3, 4, 8)
    k = torch.randn(3, 2, 8)
    positions = torch.zeros(3, dtype=torch.long)

    q_out, k_out = rope(q, k, positions)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_rope_shape_is_preserved():
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    positions = torch.arange(5)

    q_out, k_out = rope(q, k, positions)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
