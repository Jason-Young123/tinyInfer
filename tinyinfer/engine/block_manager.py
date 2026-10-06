from collections import deque
from collections import OrderedDict
import xxhash

from tinyinfer.engine.sequence import Sequence

# 当前仅实现“正在被其他活跃 Sequence 持有时可共享的 Prefix KV”, 而不是“请求结束后依然可以保留、按 LRU 等策略等待未来复用的 Prefix Cache”
# 后续待改进: 1. 真正把物理block作为cache形式, 引入替换策略, 而不是某个请求推理结束直接release
#            2. 加入automatic prefix caching逻辑

# 注意block_manager仅管理block的metadata, 不涉及KV cache实际数据
class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id         # 物理 KV block 编号
        self.ref_count = 0               # 当前有多少 Sequence 引用该 block
        self.hash: int | None = None     # 完整 token block 的哈希，用于 Prefix Cache 查找
        self.parent_hash: int = 0        # parent block的hash, 用当前hash + parent hash + token内容基本上可以hash collision
        self.token_ids: tuple[int, ...] = ()  # 该 block 对应的 token ids，仅作 metadata/哈希校验

    def bind(self, token_ids: list[int], block_hash: int, parent_hash: int):
        self.token_ids = tuple(token_ids)
        self.hash = block_hash
        self.parent_hash = parent_hash

    def reset(self):
        self.ref_count = 0
        self.hash = None
        self.parent_hash = 0
        self.token_ids = ()



def block_state(self, block_id):
    block = self.blocks[block_id]

    if block.ref_count > 0:
        return "active"

    if block.hash is not None:
        return "cached"

    return "empty"




# 所有Block的管理池
# 最重要的两个数据: free_lru, hash_to_block_ids
# 所有block只有两个状态: persistent(ref_count = 0, 包含初始态) 和 active (ref_count > 0)
# free_lru 就是一个有顺序的存放所有 persistent block的字典;
# hash_to_block_ids 存放 hash -> [block_ids] 的匹配信息, 但是其中的block既可能处于persistent状态, 也可能处于active状态
# _allocate_block: 从free_lru头部pop item, 将对应的block从persistent变为active状态, 且前后hash值发生改变;
#    注: 这个函数本身只负责删除hash_to_block_ids中原来的记录, 具体新纪录的增加和block.hash的改变(block.bind)需要在postprocess中调用cache_computed_full_blocks
# _reactivate_block: 从free_lru任意位置pop item并将对应的符合条件的block从persistent变为active状态, 或者不改动free_lru、针对已经active的block将其ref_count + 1, 且前后hash值不变
# _release_block: 向free_lru尾部push item, 将对应的block从active变为persistent状态/把原本就active的block的ref_count +, 但暂不改变hash值

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int): # 一共存在多少个物理block, 每个物理block有多大(逻辑大小, 即存放多少token对应的KV)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)] # 所有 physical KV block 的 metadata；block_id = 0...num_blocks-1
        self.free_ids = deque(range(num_blocks)) # 当前空闲 physical block id列表，allocate 从这里取，free 后再放回来
        # self.hash_to_block_ids: dict[int, int] = {} # Prefix Cache 索引：block hash -> physical block id, 即每个物理block都有唯一的hash值
        # 存在于free_lru的前提: ref_count = 0, 包含两类: 初始化的Block; 所有曾被分配过hash但现在ref_count归零的block
        self.free_lru: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks)) # 顺序字典, 用于实现LRU替换算法
        self.hash_to_block_ids: dict[int, set[int]] = {}
        # 性能统计, for future use
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
    
    def stats(self):
        return {
            "free_blocks": self.num_free_blocks,
            "cached_blocks": sum(1 for block in self.blocks if block.hash is not None),
            "active_blocks": sum(1 for block in self.blocks if block.ref_count > 0),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
        }

    # 检查映射一致性
    def check_consistency(self):
        free_set = set(self.free_lru.keys())
        for block in self.blocks:
            if block.ref_count == 0:
                assert block.block_id in free_set
            else:
                assert block.block_id not in free_set

            if block.hash is not None:
                ids = self.hash_to_block_ids.get(block.hash, set())
                assert block.block_id in ids
                # 如果你的设计规定只有完整 block 才能注册 cache
                assert len(block.token_ids) == self.block_size
            else:
                # 没有 hash 既可能是空 block, 也可能是尚未填满的 active partial block
                assert len(block.token_ids) <= self.block_size

        for block_hash, ids in self.hash_to_block_ids.items():
            for block_id in ids:
                block = self.blocks[block_id]
                assert block.hash == block_hash
                # 这里 ref_count 可以是 0, 代表 persistent cached-free block

    @property
    def num_free_blocks(self):
        return len(self.free_lru)

    # 关键的分配hash逻辑
    @staticmethod
    def _hash_block(prev_hash: int, token_ids: list[int]) -> int: # 基于前序block和当前block内的token生成综合后的hash值
        h = xxhash.xxh64()
        h.update(prev_hash.to_bytes(8, "little", signed=False)) # 先融入前序block
        for token in token_ids: # 然后遍历当前block内的token, 逐步迭代hash值
            h.update(int(token).to_bytes(8, "little", signed=True))
        return h.intdigest()

    
    ### allocate: persistent block -> active block
    ### 分配新block时的完整链条: 从free_lru头部popitem -> 检查是否为Persistent block -> 如果是, 则将其evict
    ### -> 把对应的block_id从hash_to_block_ids中移除并reset
    # 从hash_to_block_ids这个dict中删除某个Block对应的id
    def _remove_cache_index(self, block: Block): 
        if block.hash is None: # 如果该block还没被分配hash(例如block未满), 直接忽略
            return
        ids = self.hash_to_block_ids.get(block.hash)
        if ids is None: # dict中的某个hash存在但是对应的block_id为空, 一般不会发生, 防御性写法
            return
        ids.discard(block.block_id) # 从block_id这个set中去除特定的block_id
        if not ids:
            self.hash_to_block_ids.pop(block.hash, None) # 如果删除了最后一个block_id, 那么该hash值也不会再存在于dict中

    # 从hash_to_block_ids驱逐某个block id
    def _evict(self, block: Block):
        if block.ref_count != 0: # 确保是persistent cache block(即ref_count = 0)才会真正驱逐
            raise RuntimeError("cannot evict active block")
        self._remove_cache_index(block)
        block.reset() # 准备重新分配

    # 从free_lru头部弹出一个Least Recently Used的block
    def _allocate_block(self) -> Block:
        if not self.free_lru: # 没有可用的block
            raise RuntimeError("KV cache exhausted")
        block_id, _ = self.free_lru.popitem(last=False) # 从开头弹出free_lru中的item
        block = self.blocks[block_id]
        if block.ref_count != 0: # 不符合free_lru的定义, 报错
            raise RuntimeError("free-LRU corruption")
        # 如果它原来是 persistent cached block，直到此刻才真正 eviction
        if block.hash is not None:
            self._evict(block)
        block.ref_count = 1
        return block

    ### release: active block -> persistent block
    # 向free_lru尾部插入最新一个被释放的block
    def _release_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError("block refcount underflow")
        block.ref_count -= 1
        if block.ref_count == 0:
            # 内容仍然有效，只是进入可被 allocator 复用的 LRU 队列, 且追加在尾部
            self.free_lru[block_id] = None

    ### re-activate, 即位于free_lru中的block重新进入被引用状态
    def _activate_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count == 0:
            if block_id not in self.free_lru:
                raise RuntimeError("cached-free block missing from free LRU")
            self.free_lru.pop(block_id)
        block.ref_count += 1


    # 某个请求推理结束后逆序释放：tail block 通常复用价值更低，应该更早进入 LRU 老端。
    def free(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._release_block(block_id)
        seq.block_table.clear()

    ### allocate和free都是针对完整sequence进行多block分配(prefill阶段)与释放
    # 调用_take_free_block进行分配
    def allocate(self, seq: Sequence): # 为某个sequence分配新的物理block(可以分配多个)
        need = seq.num_blocks - len(seq.block_table)
        if need < 0:
            raise RuntimeError("sequence has more physical blocks than logical blocks")
        if need > self.num_free_blocks:
            raise RuntimeError("not enough KV blocks")
        for _ in range(need):
            block = self._take_free_block()
            seq.block_table.append(block.block_id)



    

    # 只读函数, 寻找某个seq有多少命中的prefix cache块, 不会修改block属性
    def find_cached_prefix(self, seq: Sequence, max_cache_hit_tokens: int | None = None) -> tuple[list[int], int, int]:
        cached_block_ids = []
        prev_hash = 0 # 第一个逻辑block没有前序block, 因此hash值为0
        num_prefix_cached_tokens = 0

        # 这里需要设置admission阶段允许命中的token上限, 为prompt_len - 1, 否则会出现admission阶段所有prompt都命中prefix cache时无法推理得到下一个token的bug
        if max_cache_hit_tokens is None:
            max_cache_hit_tokens = seq.num_prompt_tokens - 1
        max_full_blocks = max_cache_hit_tokens // self.block_size
        full_blocks = min(seq.num_prompt_tokens // self.block_size, max_full_blocks)
        #full_blocks = seq.num_tokens // self.block_size # 这个sequence具有多少个完整的block(因为prefix cache只cache完整block)

        for logical_idx in range(full_blocks): 
            token_ids = seq.block_token_ids(logical_idx) # 基于该seq内的逻辑block idx找到token_ids列表
            block_hash = self._hash_block(prev_hash, token_ids) # 确定当前block的最终hash值
            candidate_ids = self.hash_to_block_ids.get(block_hash, set()) # 注意可能一个hash会匹配到多个block_id
            matched_id = None
            for block_id in candidate_ids: # 去set(block_ids)中去匹配
                block = self.blocks[block_id]
                if (block.parent_hash == prev_hash and block.token_ids == tuple(token_ids)):
                    matched_id = block_id
                    break
            if matched_id is None: # 匹配失败
                break
            cached_block_ids.append(matched_id)
            num_prefix_cached_tokens += self.block_size
            prev_hash = block_hash # 更新prev_hash

        return cached_block_ids, num_prefix_cached_tokens, prev_hash

    # 仅在scheduler的admission步骤调用一次, 会把已命中的prefix cache block id加入block_table, 同时预分配完整prompt所需block并写入block_table;
    # 可以发现, 该函数返回后, block_table中的block_id可能已经存在于prefix cache中, 可能还没有、仅仅预分配
    def try_admit_with_prefix_cache(self, seq: Sequence) -> bool:
        if seq.block_table:
            raise RuntimeError("sequence already allocated")

        # Phase 1: 只查询 Prefix Cache，不修改任何 block 状态
        cached_block_ids, cached_tokens, prev_hash = self.find_cached_prefix(seq, max_cache_hit_tokens=seq.num_prompt_tokens - 1)
        need = seq.num_blocks - len(cached_block_ids)
        if need > self.num_free_blocks: # KV cache用尽, 无法继续分配
            return False

        # Phase 2: 资源确认足够后，才真正提交 allocation
        for block_id in cached_block_ids:
            self._activate_block(block_id) # 激活已经匹配到的prefix cache
        seq.block_table.extend(cached_block_ids)

        for _ in range(need): # 从free_lru新分配block
            block = self._allocate_block()
            seq.block_table.append(block.block_id)

        seq.num_cached_tokens = cached_tokens
        seq.num_computed_tokens = cached_tokens
        seq.last_block_hash = prev_hash
        return True

    # 在一轮推理结束后调用, 无论是decode还是chunked prefill
    def cache_computed_full_blocks(self, seq: Sequence, upto_token: int):
        full_blocks = upto_token // self.block_size

        prev_hash = 0
        for logical_idx in range(full_blocks):
            block_id = seq.block_table[logical_idx]
            block = self.blocks[block_id]
            token_ids = seq.block_token_ids(logical_idx)

            block_hash = self._hash_block(prev_hash, token_ids)

            # 已经是相同 cache identity，则跳过重复注册
            if (
                block.hash == block_hash
                and block.parent_hash == prev_hash
                and block.token_ids == tuple(token_ids)
            ):
                prev_hash = block_hash
                continue

            # 当前物理块理论上在这一刻第一次“内容稳定”; 如果此前有其他 cache identity，先清旧索引。
            self._remove_cache_index(block)
            block.bind(token_ids, block_hash, prev_hash) # 正式绑定
            self.hash_to_block_ids.setdefault(block_hash, set()).add(block_id) # 原本仅删除就记录, 此时写入新记录
            prev_hash = block_hash

        seq.last_block_hash = prev_hash

    # 预分配block; 这里的token_position = num_computed_tokens
    def ensure_capacity_for_token_position(self, seq: Sequence, token_position: int) -> bool:
        required_blocks = token_position // self.block_size + 1
        missing = required_blocks - len(seq.block_table)
        if missing <= 0:
            return True
        if missing > self.num_free_blocks:
            return False

        for _ in range(missing):
            block = self._allocate_block()
            seq.block_table.append(block.block_id)
        return True



    # 由postprocess函数调用, 目的是在上一个block恰好填充满后开辟一个新的block
    def prepare_next_decode_block(self, seq):
        needed = seq.num_blocks - len(seq.block_table)
        if needed == 0:
            return True
        if needed != 1:
            raise RuntimeError("Currently one token decode should add at most one block")
        if self.num_free_blocks == 0:
            return False
        block = self._take_free_block()
        seq.block_table.append(block.block_id)
        return True
