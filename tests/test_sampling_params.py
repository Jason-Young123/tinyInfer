import pytest
from tinyinfer import SamplingParams


def test_sampling_params():
    p = SamplingParams(temperature=0.6, max_tokens=8)
    assert p.temperature == 0.6
    assert p.max_tokens == 8


def test_reject_zero_temperature():
    with pytest.raises(ValueError):
        SamplingParams(temperature=0.0)
