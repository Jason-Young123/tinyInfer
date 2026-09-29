import torch

from tinyinfer.layers.linear import (
    LinearBase,
    MergedGateUpLinear,
    QKVParallelLinear,
)


def test_linear_base_shape():
    layer = LinearBase(10, 24, bias=False)
    x = torch.randn(5, 10)
    y = layer(x)
    assert y.shape == (5, 24)


def test_qkv_parallel_linear_shapes():
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    layer = QKVParallelLinear(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        bias=False,
    )

    x = torch.randn(7, hidden_size)
    packed = layer(x)
    q, k, v = layer.split_qkv(packed)

    assert packed.shape == (7, 64)  # 32 + 16 + 16
    assert q.shape == (7, 32)
    assert k.shape == (7, 16)
    assert v.shape == (7, 16)



def test_merged_gate_up_linear_shapes():
    layer = MergedGateUpLinear(
        hidden_size=10,
        intermediate_size=24,
        bias=False,
    )

    x = torch.randn(5, 10)
    packed = layer(x)
    gate, up = layer.split_gate_up(packed)

    assert packed.shape == (5, 48)
    assert gate.shape == (5, 24)
    assert up.shape == (5, 24)










