from tinyinfer.engine.sequence import Sequence, SequenceStatus
from tinyinfer import SamplingParams


def test_sequence_lifecycle():
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0

    seq.append_token(13)
    assert seq.num_completion_tokens == 1
    assert not seq.should_stop(-1)

    seq.append_token(14)
    assert seq.should_stop(-1)
