from tinyinfer import SamplingParams
from tinyinfer.config import Config
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence, SequenceStatus

def make_seq(n_prompt, n_out):
    return Sequence(
        [1] * n_prompt,
        SamplingParams(temperature=1.0, max_tokens=n_out, ignore_eos=True),
    )


def make_config(**kwargs):
    base = dict(
        max_num_seqs=8,
        max_num_batched_tokens=8,
        max_model_len=128,
        kvcache_block_size=4,
        num_kvcache_blocks=64,
    )
    base.update(kwargs)
    return Config(**base)


def test_mixed_batch_contains_decode_and_prefill():
    sched = Scheduler(make_config(max_num_batched_tokens=6))

    running = make_seq(4, 4)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = 4
    running.block_table = [sched.block_manager._allocate_block().block_id]
    running.append_token(99)  # 99 尚未计算 KV，因此需要 decode
    sched.running.append(running)

    waiting = make_seq(10, 2)
    sched.add(waiting)

    out = sched.schedule()

    assert any(not x.is_prefill for x in out.items)
    assert any(x.is_prefill for x in out.items)
    assert out.num_scheduled_tokens <= 6









