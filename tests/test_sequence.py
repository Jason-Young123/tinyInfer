from tinyinfer.engine.sequence import Sequence, SequenceStatus
from tinyinfer import SamplingParams


def test_sequence_lifecycle():
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 3
    assert seq.num_completion_tokens == 0

    seq.append_token(13)
    assert seq.num_completion_tokens == 1
    assert not seq.should_stop(-1, 1000)

    seq.append_token(14)
    assert seq.should_stop(-1, 1000)


def test_computed_progress_is_not_cache_hit_count():
    seq = Sequence([10, 11, 12, 13, 14, 15], SamplingParams(max_tokens = 2))
    seq.num_cached_tokens = 2
    seq.num_computed_tokens = 4

    assert seq.num_cached_tokens == 2
    assert seq.num_computed_tokens == 4
    assert seq.num_prompt_tokens_remaining == 2


def test_sampled_token_has_no_kv_until_next_forward():
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    seq.num_computed_tokens = 3

    seq.append_token(13)

    assert seq.num_tokens == 4
    assert seq.num_computed_tokens == 3
    assert seq.needs_decode




