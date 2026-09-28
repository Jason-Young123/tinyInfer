from tinyinfer import SamplingParams
from tinyinfer.engine.block_manager import BlockManager
from tinyinfer.engine.sequence import Sequence

"""
def test_allocate_and_free():
    Sequence.block_size = 4
    seq = Sequence(list(range(9)), SamplingParams())
    mgr = BlockManager(num_blocks=8, block_size=4)

    mgr.allocate(seq)
    assert len(seq.block_table) == 3
    assert mgr.num_free_blocks == 5

    mgr.free(seq)
    assert seq.block_table == []
    assert mgr.num_free_blocks == 8
"""

"""
def test_prefix_cache_share_full_block():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1,2,3,4,9], SamplingParams())
    mgr.allocate_with_prefix_cache(a)
    first = a.block_table[0]

    b = Sequence([1,2,3,4,8], SamplingParams())
    mgr.allocate_with_prefix_cache(b)

    assert b.num_cached_tokens == 4
    assert b.block_table[0] == first
    assert mgr.blocks[first].ref_count == 2
"""

"""def test_partial_block_not_cached():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(a)

    b = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(b)

    assert b.num_cached_tokens == 0
"""


def test_finished_request_keeps_cached_block():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1, 2, 3, 4, 9], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(a)
    a.num_computed_tokens = a.num_prompt_tokens
    mgr.cache_computed_full_blocks(a, a.num_computed_tokens)

    cached_block = a.block_table[0]
    mgr.free(a)

    block = mgr.blocks[cached_block]
    assert block.ref_count == 0
    assert block.hash is not None
    assert block.token_ids == (1, 2, 3, 4)



def test_reuse_cache_after_original_request_finished():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1, 2, 3, 4, 9], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(a)
    a.num_computed_tokens = a.num_prompt_tokens
    mgr.cache_computed_full_blocks(a, a.num_computed_tokens)
    first = a.block_table[0]
    mgr.free(a)

    b = Sequence([1, 2, 3, 4, 8], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(b)

    assert b.num_cached_tokens == 4
    assert b.block_table[0] == first
    assert mgr.blocks[first].ref_count == 1




def test_lru_evicts_cached_free_block_only_when_reused():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=2, block_size=4)

    a = Sequence([1, 2, 3, 4], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(a)
    a.num_computed_tokens = 4
    mgr.cache_computed_full_blocks(a, 4)
    old_hash = mgr.blocks[a.block_table[0]].hash
    mgr.free(a) # 逐一进行_release_block

    assert old_hash in mgr.hash_to_block_ids

    # 分配其他内容，最终需要真正拿走 cached-free block。
    c = Sequence([9, 9, 9, 9, 8], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(c)

    # 至少被重新分配的旧块必须已不再保留旧 cache identity。
    for block_id in c.block_table:
        block = mgr.blocks[block_id]
        assert not (
            block.hash == old_hash and block.token_ids == (1, 2, 3, 4)
        )



def test_full_prompt_cache_hit_keeps_one_block_for_logits():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1,2,3,4,5,6,7,8], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(a)
    a.num_computed_tokens = 8
    mgr.cache_computed_full_blocks(a, 8)
    mgr.free(a)

    b = Sequence([1,2,3,4,5,6,7,8], SamplingParams())
    assert mgr.try_admit_with_prefix_cache(b)

    # block granularity 下最多跳过前 4 个，最后 4 个重算以获得 logits。
    assert b.num_cached_tokens == 4
    assert b.num_computed_tokens == 4










