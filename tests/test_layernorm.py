import torch

from tinyinfer.layers.layernorm import RMSNorm


def test_rmsnorm_shape_and_finite():
    norm = RMSNorm(16, eps=1e-6)
    x = torch.randn(3, 5, 16)
    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rmsnorm_matches_manual_reference():
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.fill_(1.0)

    x = torch.randn(4, 8)
    y = norm(x)

    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref.to(x.dtype)

    torch.testing.assert_close(y, ref)
