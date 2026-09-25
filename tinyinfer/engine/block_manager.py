from collections import deque
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
        self.token_ids: tuple[int, ...] = ()  # 该 block 对应的 token ids，仅作 metadata/哈希校验

    def bind(self, token_ids: list[int], block_hash: int | None):
        self.token_ids = tuple(token_ids) # 记录此物理 block 当前代表哪些 token
        self.hash = block_hash            # 绑定其 prefix-cache, 便于后续查找

    def reset(self):
        self.ref_count = 0                # 无 Sequence 引用
        self.hash = None                  # 清除旧 prefix-cache 身份
        self.token_ids = ()               # 清除旧 token metadata



# 所有Block的管理池
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int): # 一共存在多少个物理block, 每个物理block有多大(逻辑大小, 即存放多少token对应的KV)
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)] # 所有 physical KV block 的 metadata；block_id = 0...num_blocks-1
        self.free_ids = deque(range(num_blocks)) # 当前空闲 physical block id列表，allocate 从这里取，free 后再放回来
        self.hash_to_block_id: dict[int, int] = {} # Prefix Cache 索引：block hash -> physical block id, 即每个物理block都有唯一的hash值

    # 检查映射一致性
    def check_consistency(self):
        free_set = set(self.free_ids)
        for block in self.blocks:
            if block.block_id in free_set:
                assert block.ref_count == 0
            else:
                assert block.ref_count > 0
        for block_hash, block_id in self.hash_to_block_id.items():
            block = self.blocks[block_id]
            assert block.hash == block_hash
            assert block.ref_count > 0

    @property
    def num_free_blocks(self) -> int: # 还有多少剩余可分配的物理block
        return len(self.free_ids)

    def _take_free_block(self) -> Block: # 从free_ids列表中新分配一个block
        if not self.free_ids:
            raise RuntimeError("KV cache exhausted")  # 没有可用物理 block
        block_id = self.free_ids.popleft()            # FIFO 取一个空闲 block id
        block = self.blocks[block_id]                 # 找到对应 Block metadata
        if block.ref_count != 0:
            raise RuntimeError("free-list corruption")# free list 中的 block 理应无人引用
        block.ref_count = 1                           # 新 Sequence 成为第一个引用者
        return block

    ## 这里可能发生hash collide, 但实际上不会错误reset block;
    ##   比如block7和block11发生了hash collide, 且hash_to_block_id中Hash_Collided指向的是block7;
    ##   那么假设block7所对应的推理请求先结束, 那么其会调用_release_block(7), 删除HC -> block7这条记录并把block7 reset, 但不会动block11;
    ##   假设block11对应的请求先结束, 那么其会调用_release_block(11), 删除HC -> block7并把block11 reset, 但不会动block7;
    ##   也就是说, _release_block可能错误删除了字典中的HC->blocki的信息但不会影响正在推理中的请求, 只不过当新请求进来时, 可能影响其prefix cache的命中率
    def _release_block(self, block_id: int): # _take_free_block的逆操作, 将指定编号的物理block减少一次引用, 并做可能的回收
        block = self.blocks[block_id]                  # 找到待释放 block
        if block.ref_count <= 0:
            raise RuntimeError("block refcount underflow")  # 防止重复释放
        block.ref_count -= 1                           # 当前 Sequence 放弃引用
        if block.ref_count == 0:
            # 改进: 只有全局 Prefix Cache 索引确实指向当前 block, 才能删除 hash -> block_id 映射。
            if (
                block.hash is not None
                and self.hash_to_block_id.get(block.hash) == block_id
            ):
                self.hash_to_block_id.pop(block.hash)
            block.reset()
            self.free_ids.append(block_id)

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

    # 调用_release_block进行释放
    def free(self, seq: Sequence): # 对于某个sequence分配的所有物理block进行占用释放(可以释放多个)
        for block_id in seq.block_table:
            self._release_block(block_id)
        seq.block_table.clear()

    # 关键的分配hash逻辑
    @staticmethod
    def _hash_block(prev_hash: int, token_ids: list[int]) -> int: # 基于前序block和当前block内的token生成综合后的hash值
        h = xxhash.xxh64()
        h.update(prev_hash.to_bytes(8, "little", signed=False)) # 先融入前序block
        for token in token_ids: # 然后遍历当前block内的token, 逐步迭代hash值
            h.update(int(token).to_bytes(8, "little", signed=True))
        return h.intdigest()

    ## 这个函数内部发生hash collide的情况: 
    ##   某个block找到了匹配的hash->block_id记录, 
    ##   但实际上发生了 hash(prefixA + [1,2,3,4]) == hash(prefixB + [1,2,3,4])的极小概率事件;
    ##   此时理论上这个命中的block不完全匹配, 不应该使用, 但此处除非逐次比较所有前序hash(tokens to be exactly)否则根本无法判断发生了collide;
    ##   鉴于判断复杂程度和自身的极小概率性, 此处就忽略这种collide;
    ##   产生的后果是: 实际不匹配的KV cache误认为匹配, 导致这个请求的上下文出错、回复有问题 
    # 用于判断当前seq到底有多少token/逻辑block已经作为prefix cache存在于物理block中, 并返回最后一个命中的prefix cache的hash值
    def find_cached_prefix(self, seq: Sequence) -> tuple[list[int], int, int]: # 只读函数, 寻找某个seq有多少命中的prefix cache块, 不会修改block属性
        cached_ids = []
        prev_hash = 0 # 第一个逻辑block没有前序block, 因此hash值为0
        num_prefix_cached_tokens = 0
        full_blocks = seq.num_tokens // self.block_size # 这个sequence具有多少个完整的block(因为prefix cache只cache完整block)

        for logical_idx in range(full_blocks): 
            token_ids = seq.block_token_ids(logical_idx) # 基于该seq内的逻辑block idx找到token_ids列表
            block_hash = self._hash_block(prev_hash, token_ids) # 确定当前block的最终hash值
            block_id = self.hash_to_block_id.get(block_hash) # 去已经被分配的物理block中去寻找是否有匹配的hash值
            if block_id is None: # 匹配失败, 直接停止, 因为后续必然会失败
                break
            block = self.blocks[block_id] # 成功匹配到了物理block
            if block.token_ids != tuple(token_ids): # 再次确认目标物理block内部token是否和当前逻辑block内token相同
                break
            # 确定匹配成功
            # block.ref_count += 1
            cached_ids.append(block_id)
            num_prefix_cached_tokens += self.block_size
            prev_hash = block_hash # 更新prev_hash

        return cached_ids, num_prefix_cached_tokens, prev_hash

    # 仅用于处理prefill workload, 对整段prompt进行prefix cache注册/分配
    # 这个函数同样可能发生hash collide, 发生在新分配block的过程中:
    ##   该新分配block产生的hash和某个已分配block的hash值相同; 此时该请求会错误地使用其他请求或自身的历史KV cache
    def try_allocate_with_prefix_cache(self, seq: Sequence) -> bool:
        if seq.block_table:
            raise RuntimeError("sequence already allocated")

        # Phase 1: 只查询 Prefix Cache，不修改任何 block 状态
        cached_ids, cached_tokens, prev_hash = self.find_cached_prefix(seq)
        need = seq.num_blocks - len(cached_ids) # 还需分配的block数目
        if need > self.num_free_blocks: # 如果考虑了prefix cache之后空间还是不够则直接返回False
            return False

        # Phase 2: 资源确认足够后，才真正提交 allocation
        # cached blocks 现在正式被当前 Sequence 引用, 因此ref_count + 1
        for block_id in cached_ids:
            self.blocks[block_id].ref_count += 1
        seq.block_table.extend(cached_ids) # 将命中prefix cache的块注册到block_table中, 后续无需再分配
        seq.num_prefix_cached_tokens = cached_tokens # 已经存在prefix cache中的token总数
        
        self.allocate(seq) # 为后面未命中prefix cache的部分再进行分配
        seq.last_block_hash = prev_hash

        # Phase 3: 注册新产生的完整 Prefix Cache blocks
        # 注意, 为了避免同一批prefill任务中存在相同prefix、B提前访问A实际上还没有算出来的KV cache问题,
        # 需要将prefix cache注册逻辑单独分离为cache_computed_full_blocks函数, 并在postprocess函数中对其进行调用
        #start_idx = len(cached_ids)
        #for logical_idx in range(start_idx, len(seq.block_table)):
        #    block_id = seq.block_table[logical_idx]
        #    token_ids = seq.block_token_ids(logical_idx)
        #    # Prefix Cache 只注册完整 block
        #    if len(token_ids) != self.block_size:
        #        break
        #    block_hash = self._hash_block(prev_hash, token_ids)     # 分配该block的hash
        #    block = self.blocks[block_id]                           # 获取block
        #    block.bind(token_ids, block_hash)                       # 为该block绑定token和hash信息
        #    self.hash_to_block_id.setdefault(block_hash, block_id)  # 更新hash_to_block_id这个dict, 如果hash已经存在则跳过更新
        #    prev_hash = block_hash
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

    # 由postprocess函数调用, 目的是为所有已经full但尚未注册prefix cache的block进行注册
    def cache_computed_full_blocks(self, seq: Sequence):
        num_tokens = seq.num_tokens # 注意这里的num_tokens在刚进入postprocess函数时还没更新, 仍然是推理前的总token数(不包括才由Model推理得到的token)
        num_prefix_cached_tokens = seq.num_prefix_cached_tokens # 已经存在于prefix cache中的token数
        cached_blocks = (num_prefix_cached_tokens // self.block_size) # 已经注册到cache中的完整block数量
        computed_blocks = (num_tokens // self.block_size) # 当前已经计算完成的完整block数量

        # 没有新增完整block, 直接返回
        if computed_blocks <= cached_blocks:
            return

        # 从第一个未cache block开始注册
        prev_hash = seq.last_block_hash
        for logical_idx in range(cached_blocks,computed_blocks):
            block_id = seq.block_table[logical_idx] # logical block对应physical block
            block = self.blocks[block_id]
            token_ids = seq.block_token_ids(logical_idx)
            if len(token_ids) != self.block_size: # 理论检查
                raise RuntimeError("only full blocks can enter prefix cache")
            block_hash = self._hash_block(prev_hash,token_ids) # 生成链式hash
            block.bind(token_ids, block_hash) # 绑定metadata
            self.hash_to_block_id.setdefault(block_hash, block_id) # 注册全局prefix索引
            prev_hash = block_hash # 更新链
            seq.num_prefix_cached_tokens += self.block_size # 更新num_prefix_cached_tokens
        
        seq.last_block_hash = prev_hash # 更新last_block_hash


