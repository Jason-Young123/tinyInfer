from tinyinfer import SamplingParams
from tinyinfer.config import Config
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence, SequenceStatus


def make_seq(n_prompt, n_out):
    return Sequence(
        [1] * n_prompt,
        SamplingParams(temperature=1.0, max_tokens=n_out, ignore_eos=True),
    )


def test_max_num_seqs():
    sched = Scheduler(Config(max_num_seqs=2, max_num_batched_tokens=1000))
    seqs = [make_seq(10, 2) for _ in range(3)]
    for s in seqs:
        sched.add(s)

    picked, is_prefill = sched.schedule()
    assert is_prefill
    assert len(picked) == 2
    assert len(sched.running) == 2
    assert len(sched.waiting) == 1
    assert all(s.status is SequenceStatus.RUNNING for s in picked)


def test_token_budget():
    sched = Scheduler(Config(max_num_seqs=8, max_num_batched_tokens=25))
    sched.add(make_seq(20, 2))
    sched.add(make_seq(20, 2))

    picked, _ = sched.schedule()
    assert len(picked) == 1
