import torch

from tinyinfer.layers.sampler import Sampler


def test_sampler_greedy():
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 2.0, 0.3],
            [4.0, 1.0, 3.0],
        ]
    )
    temperatures = torch.ones(2)
    greedy = torch.tensor([True, True])

    out = sampler(logits, temperatures, greedy)
    assert out.tolist() == [1, 0]


def test_sampler_mixed_greedy_and_sampling_shapes():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.randn(4, 32)
    temperatures = torch.tensor([1.0, 0.8, 1.0, 0.7])
    greedy = torch.tensor([True, False, True, False])

    out = sampler(logits, temperatures, greedy)
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert 0 <= int(out.min())
    assert int(out.max()) < 32
