import pytest
import torch
import torch.nn.functional as F

from tinyinfer.layers.flash_attention import flash_attention


CUDA_AVAILABLE = torch.cuda.is_available()


def torch_reference(q, k, v, query_start_pos, scale):
    hq = q.shape[1]
    hkv = k.shape[1]

    q_ref = q.transpose(0, 1).unsqueeze(0)
    k_ref = k.transpose(0, 1).unsqueeze(0)
    v_ref = v.transpose(0, 1).unsqueeze(0)

    if hq != hkv:
        repeat = hq // hkv
        k_ref = k_ref.repeat_interleave(repeat, dim=1)
        v_ref = v_ref.repeat_interleave(repeat, dim=1)

    tq = q.shape[0]
    tk = k.shape[0]

    q_pos = torch.arange(
        query_start_pos,
        query_start_pos + tq,
        device=q.device,
    )
    k_pos = torch.arange(tk, device=q.device)

    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    causal = causal.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=causal,
        is_causal=False,
        scale=scale,
    )

    return out.squeeze(0).transpose(0, 1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tq,tk,query_start_pos",
    [
        (37, 37, 0),
        (7, 37, 30),
        (1, 37, 36),
    ],
)
def test_flash_attention_matches_torch(dtype, tq, tk, query_start_pos):
    torch.manual_seed(0)

    hq = 8
    hkv = 2
    head_dim = 128
    scale = head_dim ** -0.5

    q = torch.randn(
        tq,
        hq,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1
    k = torch.randn(
        tk,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1
    v = torch.randn(
        tk,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1

    expected = torch_reference(
        q,
        k,
        v,
        query_start_pos=query_start_pos,
        scale=scale,
    )
    actual = flash_attention(
        q,
        k,
        v,
        query_start_pos=query_start_pos,
        scale=scale,
        is_causal=True # 启用causal mask
    )

    if dtype == torch.bfloat16:
        atol = 0.10
        rtol = 0.10
    else:
        atol = 0.06
        rtol = 0.06

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )





