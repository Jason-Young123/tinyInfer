import torch

from tinyinfer.layers.activation import SiluAndMul


def test_silu_and_mul_shape():
    x = torch.randn(7, 2 * 16)
    y = SiluAndMul()(x)
    assert y.shape == (7, 16)


def test_silu_and_mul_matches_reference():
    x = torch.randn(5, 48)
    gate, up = x.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up

    y = SiluAndMul()(x)
    torch.testing.assert_close(y, ref)
