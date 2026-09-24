from tinyinfer import SamplingParams
from tinyinfer.engine.block_manager import BlockManager
from tinyinfer.engine.sequence import Sequence


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


def test_partial_block_not_cached():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(a)

    b = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(b)

    assert b.num_cached_tokens == 0
