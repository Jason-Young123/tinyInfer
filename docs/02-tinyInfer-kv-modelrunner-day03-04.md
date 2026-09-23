# tinyInfer 手把手教程（中）：Day 03–04 —— Paged KV Cache、Prefix Cache、Attention Metadata 与 ModelRunner

> 前置：已经完成上篇 Day01–02，拥有 `Config / SamplingParams / Sequence / Scheduler / LLMEngine / ToyModelRunner`，能够在 CPU 上验证 continuous batching 控制面。
>
> 本篇目标：把 tinyInfer 从“会调度请求”推进到“知道每个 token 的 KV 应该存到哪里，并把 Sequence 编译成 GPU/模型可消费的 tensor metadata”。

---

# Day 03：从“request state”进入“paged KV memory state”

# 0. 今天最重要的心智模型

如果你仍然把 KV cache 理解为：

```text
每个 request 自己有一个 [max_seq_len, ...] 大 tensor
```

那么你还没有理解 vLLM 的核心。

我们今天要做的是：

```text
全局 KV physical block pool
        ↓
每条 Sequence 只保存 logical block → physical block 的映射
        ↓
真正 token 的写入位置由 slot_mapping 决定
```

这不是单纯的数据结构优化，而是服务端高并发推理能否把显存利用起来的关键。

---

# 1. Step 0：先只增加两个文件

今天先新增：

```text
tinyinfer/engine/block_manager.py
tests/test_block_manager.py
```

并修改：

```text
tinyinfer/engine/scheduler.py
tinyinfer/engine/sequence.py
```

不要碰 attention/model。

---

# 2. Step 1：先算一遍 KV cache 显存

对 decoder-only transformer，一层每 token roughly 需要：

```text
K: num_kv_heads × head_dim
V: num_kv_heads × head_dim
```

所有层：

```text
bytes/token
≈ num_layers
× 2
× num_kv_heads
× head_dim
× dtype_bytes
```

如果：

```text
28 layers
8 KV heads
128 head dim
BF16 = 2 bytes
```

则：

```text
28 × 2 × 8 × 128 × 2
≈ 114688 bytes/token
≈ 112 KB/token
```

数千 token × 数十并发后，KV 很快成为主要显存占用之一。

这就是为什么我们不能用粗暴的：

```text
request × max_model_len
```

静态预留。

---

# 3. Step 2：先区分四层地址

假设：

```text
block_size = 4 tokens
```

某 seq token positions：

```text
0 1 2 3 | 4 5 6 7 | 8
```

四个概念：

```text
Token position:
8

Logical block index:
8 // 4 = 2

Physical block id:
seq.block_table[2] = 11

Physical slot:
11 * 4 + (8 % 4) = 44
```

以后 `slot_mapping` 本质上就是：

```text
each scheduled token
    ↓
its physical KV slot
```

---

# 4. Step 3：给 Sequence 增加 block 相关帮助函数

修改：

```text
tinyinfer/engine/sequence.py
```

加入：

```python
@property
def num_blocks(self) -> int:
    n = self.num_tokens
    return (n + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self) -> int:
    rem = self.num_tokens % self.block_size
    return rem if rem else self.block_size

def block_token_ids(self, logical_idx: int) -> list[int]:
    begin = logical_idx * self.block_size
    end = min(begin + self.block_size, self.num_tokens)
    return self.token_ids[begin:end]
```

临时测试：

```python
seq = Sequence([1,2,3,4,5,6,7,8,9], SamplingParams())
seq.block_size = 4
assert seq.num_blocks == 3
assert seq.block_token_ids(0) == [1,2,3,4]
assert seq.block_token_ids(2) == [9]
```

---

# 5. Step 4：写 `Block` —— physical page 的 metadata

创建：

```text
tinyinfer/engine/block_manager.py
```

第一部分：

```python
from collections import deque
import xxhash

from tinyinfer.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None
        self.token_ids: tuple[int, ...] = ()

    def bind(self, token_ids: list[int], block_hash: int | None):
        self.token_ids = tuple(token_ids)
        self.hash = block_hash

    def reset(self):
        self.ref_count = 0
        self.hash = None
        self.token_ids = ()
```

为什么 Block 不直接存真正 K/V tensor？

因为这里是**allocator metadata**：

```text
BlockManager
    管理 block ownership / hash / refcount / free list

ModelRunner
    真正拥有 GPU KV tensor storage
```

要把“内存管理策略”和“GPU tensor”解耦。

---

# 6. Step 5：先写没有 prefix cache 的 BlockManager

继续：

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids = deque(range(num_blocks))
        self.hash_to_block_id: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    def _take_free_block(self) -> Block:
        if not self.free_ids:
            raise RuntimeError("KV cache exhausted")
        block = self.blocks[self.free_ids.popleft()]
        if block.ref_count != 0:
            raise RuntimeError("free-list corruption")
        block.ref_count = 1
        return block

    def _release_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError("block refcount underflow")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.hash is not None:
                self.hash_to_block_id.pop(block.hash, None)
            block.reset()
            self.free_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        need = seq.num_blocks - len(seq.block_table)
        return need <= self.num_free_blocks

    def allocate(self, seq: Sequence):
        need = seq.num_blocks - len(seq.block_table)
        if need < 0:
            raise RuntimeError("sequence has more physical blocks than logical blocks")
        if need > self.num_free_blocks:
            raise RuntimeError("not enough KV blocks")

        for _ in range(need):
            block = self._take_free_block()
            seq.block_table.append(block.block_id)

    def free(self, seq: Sequence):
        for block_id in seq.block_table:
            self._release_block(block_id)
        seq.block_table.clear()
```

## 临时测试

```text
tests/test_block_manager.py
```

```python
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
```

运行：

```bash
pytest -q tests/test_block_manager.py
```

---

# 7. Step 6：为什么 decode 时可能需要 `may_append()`？

prefill 后：

```text
seq.num_tokens = 256
block_size = 256
```

它恰好填满 1 block。

采样得到一个新 token 后：

```text
seq.num_tokens = 257
```

下一轮 decode 要给 token #256 写 KV，必须新开第 2 block。

所以 BlockManager 需要：

```python
def can_append(self, seq: Sequence) -> bool:
    needed_blocks = seq.num_blocks - len(seq.block_table)
    return needed_blocks <= self.num_free_blocks


def may_append(self, seq: Sequence):
    needed_blocks = seq.num_blocks - len(seq.block_table)
    for _ in range(max(0, needed_blocks)):
        block = self._take_free_block()
        seq.block_table.append(block.block_id)
```

这里 `may_append` 不等于“append token”，而是：

```text
如果这个 sequence 的逻辑长度增长导致需要新的 physical block，补 allocator state
```

---

# 8. Step 7：加入 Prefix Cache —— 为什么只 hash full block？

两个 prompt：

```text
A = [system prompt 256 tokens] + user A
B = [system prompt 256 tokens] + user B
```

前 256 tokens 完全相同。

如果 block size=256，则 A/B 第一块 K/V 完全相同，可以直接共享 physical block。

但如果 prefix 只有 100 tokens：

```text
block 未完整填满
```

后续 token 不同会继续写入同一个 block，因此不能把它当稳定 immutable cache entry。

所以最简单安全策略：

```text
只缓存完整 block
```

---

# 9. Step 8：实现 chaining hash

如果仅 hash 当前 block token：

```text
block #1 = [100, 200]
```

无法区分它前面接的是哪段 prefix。

因此 hash 应该包含：

```text
previous block hash + current block token ids
```

加入：

```python
@staticmethod
def _hash_block(prev_hash: int, token_ids: list[int]) -> int:
    h = xxhash.xxh64()
    h.update(prev_hash.to_bytes(8, "little", signed=False))
    for token in token_ids:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.intdigest()
```

查找 shared prefix：

```python
def find_cached_prefix(self, seq: Sequence) -> tuple[list[int], int]:
    cached_ids = []
    prev_hash = 0
    num_cached_tokens = 0

    full_blocks = seq.num_tokens // self.block_size

    for logical_idx in range(full_blocks):
        token_ids = seq.block_token_ids(logical_idx)
        block_hash = self._hash_block(prev_hash, token_ids)
        block_id = self.hash_to_block_id.get(block_hash)
        if block_id is None:
            break

        block = self.blocks[block_id]
        if block.token_ids != tuple(token_ids):
            break

        block.ref_count += 1
        cached_ids.append(block_id)
        num_cached_tokens += self.block_size
        prev_hash = block_hash

    return cached_ids, num_cached_tokens
```

真正 allocation：

```python
def allocate_with_prefix_cache(self, seq: Sequence):
    if seq.block_table:
        raise RuntimeError("sequence already allocated")

    cached_ids, cached_tokens = self.find_cached_prefix(seq)
    seq.block_table.extend(cached_ids)
    seq.num_cached_tokens = cached_tokens

    self.allocate(seq)

    # 把新分配出来且已经完整的块注册进 prefix cache。
    prev_hash = 0
    for logical_idx, block_id in enumerate(seq.block_table):
        token_ids = seq.block_token_ids(logical_idx)
        if len(token_ids) != self.block_size:
            break

        block_hash = self._hash_block(prev_hash, token_ids)
        block = self.blocks[block_id]
        if block.hash is None:
            block.bind(token_ids, block_hash)
            self.hash_to_block_id.setdefault(block_hash, block_id)
        prev_hash = block_hash
```

> 教学版这里把 lookup/register 写得直观，后续可以再处理“同一步多个请求命中刚生成 prefix”的 late merge 等更复杂情况。

---

# 10. Step 9：Prefix Cache 单元测试

补：

```python
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
```

再加 partial block 不共享：

```python
def test_partial_block_not_cached():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(a)

    b = Sequence([1,2,3], SamplingParams())
    mgr.allocate_with_prefix_cache(b)

    assert b.num_cached_tokens == 0
```

---

# 11. Step 10：把 BlockManager 接入 Scheduler

修改 Scheduler 初始化：

```python
from tinyinfer.engine.block_manager import BlockManager
```

```python
self.block_manager = BlockManager(
    num_blocks=config.num_kvcache_blocks,
    block_size=config.kvcache_block_size,
)
```

由于 Day03 还没有根据 GPU 显存动态计算 block 数，先在 Config 设置一个学习默认值：

```python
num_kvcache_blocks: int = 128
```

prefill admission 修改为：

```python
seq = self.waiting[0]

if not self.block_manager.can_allocate(seq):
    break

self.waiting.popleft()
self.block_manager.allocate_with_prefix_cache(seq)
```

完成时：

```python
self.block_manager.free(seq)
```

decode 前：

```python
if not self.block_manager.can_append(seq):
    # 最小教学版：暂停 admission；Day07 再讨论 preemption/fairness。
    continue
self.block_manager.may_append(seq)
```

此时 Scheduler 第一次真正与显存 allocator state 耦合。

---

# 12. Day03 小创新：加入 KV allocator consistency checker

这是第一个你可以明确说“在复现过程中自己加的工程改进”的地方，而且非常适合学习。

在 `BlockManager` 增加：

```python
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
```

然后在环境变量打开时：

```python
if trace_enabled("TINYINFER_CHECK_KV"):
    self.block_manager.check_consistency()
```

价值：

```text
复现系统时，不只“把功能做出来”
而是主动给 allocator 加 invariants
```

这比盲目加复杂优化更适合前期。

建议 commit：

```bash
git add .
git commit -m "day03: implement paged KV block manager and prefix cache"
```

---

# Day 04：从 Sequence/Block Table 到 ModelRunner/Attention Metadata

# 13. 今天的核心问题

Scheduler 现在能输出：

```text
Sequence objects
block_table
num_cached_tokens
num_scheduled_tokens
```

但模型 forward 需要：

```text
input_ids: Tensor
positions: Tensor
slot_mapping: Tensor
block_tables: Tensor
context_lens: Tensor
cu_seqlens_q/k: Tensor
```

所以 ModelRunner 的核心职责不是“调用 model()”这么简单，而是：

```text
runtime state
    ↓ compile/pack
GPU execution state
```

---

# 14. Step 1：创建 runtime Context

新增目录：

```bash
mkdir -p tinyinfer/utils tinyinfer/layers
touch tinyinfer/utils/__init__.py tinyinfer/layers/__init__.py
```

创建：

```text
tinyinfer/utils/context.py
```

```python
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def set_context(**kwargs):
    global _CONTEXT
    _CONTEXT = Context(**kwargs)


def get_context() -> Context:
    return _CONTEXT


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

为什么用 context，而不是给每一层 attention 显式传 8 个 metadata 参数？

因为 transformer forward 层次很深：

```text
Model
 └─ DecoderLayer
     └─ Attention
```

metadata 是一次 runtime execution 的全局上下文，不是模型权重。用 context 可避免把所有 forward 签名污染成：

```python
forward(x, positions, slot_mapping, block_tables, context_lens, ...)
```

---

# 15. Step 2：先独立实现 slot mapping 计算

创建：

```text
tests/test_attention_meta.py
```

先写帮助函数到 `model_runner.py`：

```python
def token_to_slot(seq, token_position: int) -> int:
    block_size = seq.block_size
    logical_block = token_position // block_size
    offset = token_position % block_size
    physical_block = seq.block_table[logical_block]
    return physical_block * block_size + offset
```

测试：

```python
def test_token_to_slot():
    class S:
        block_size = 4
        block_table = [7, 2, 11]

    assert token_to_slot(S(), 0) == 28
    assert token_to_slot(S(), 6) == 10
    assert token_to_slot(S(), 8) == 44
```

先把地址翻译搞对，再碰 attention。

---

# 16. Step 3：写 `prepare_prefill()`

把 Day01 的 `ToyModelRunner` 先保留，新增真实 `ModelRunner` 类：

```python
import torch

from tinyinfer.utils.context import set_context, reset_context


class ModelRunner:
    def __init__(self, config, model=None, device="cuda"):
        self.config = config
        self.model = model
        self.device = torch.device(device)

    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        slot_mapping = []
        q_lens = []
        k_lens = []

        for seq in seqs:
            start = seq.num_cached_tokens
            end = seq.num_tokens

            tokens = seq.token_ids[start:end]
            pos = list(range(start, end))

            input_ids.extend(tokens)
            positions.extend(pos)
            slot_mapping.extend(token_to_slot(seq, p) for p in pos)

            q_lens.append(len(tokens))
            k_lens.append(end)

        def prefix_sum(lengths):
            out = [0]
            for x in lengths:
                out.append(out[-1] + x)
            return out

        cu_q = prefix_sum(q_lens)
        cu_k = prefix_sum(k_lens)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, device=self.device)
        cu_k = torch.tensor(cu_k, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens, default=0),
            max_seqlen_k=max(k_lens, default=0),
            slot_mapping=slot_mapping,
        )
        return input_ids, positions
```

这里的关键不是 Python list 转 tensor，而是理解：

```text
prefix-cache hit 后：
start = num_cached_tokens

所以 prefill 并不一定从 prompt position 0 开始
```

这就是 prefix caching 与 model runner 真正连接的位置。

---

# 17. Step 4：为什么 Prefill 要 `cu_seqlens`？

如果本轮有三条不同长度 sequence：

```text
A q_len=3
B q_len=5
C q_len=2
```

将 token packed 成：

```text
[A A A B B B B B C C]
```

FlashAttention 需要知道边界：

```text
cu_seqlens_q = [0,3,8,10]
```

含义：

```text
seq0 = [0:3]
seq1 = [3:8]
seq2 = [8:10]
```

它让 variable-length batch 不必 pad 到最长序列。

这就是 continuous batching 与 varlen attention 之间的桥。

---

# 18. Step 5：写 `prepare_decode()`

Decode 每条 seq 本轮只算最后一个未缓存 token：

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    block_tables = []

    max_blocks = max(len(seq.block_table) for seq in seqs)

    for seq in seqs:
        pos = seq.num_tokens - 1
        input_ids.append(seq.last_token)
        positions.append(pos)
        slot_mapping.append(token_to_slot(seq, pos))
        context_lens.append(seq.num_tokens)

        table = list(seq.block_table)
        table.extend([-1] * (max_blocks - len(table)))
        block_tables.append(table)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
    positions = torch.tensor(positions, dtype=torch.long, device=self.device)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device)

    set_context(
        is_prefill=False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )
    return input_ids, positions
```

为什么 decode metadata 和 prefill 不同？

```text
Prefill:
    packed many query tokens
    varlen q/k boundaries

Decode:
    one query token per seq
    but must read arbitrary-length historical KV via block table
```

这就是两条 attention path 的根本差异。

---

# 19. Step 6：先实现一个可验证的 KV Cache Storage

在 `attention.py` 创建一个教学版 KV 存储器：

```python
import torch
from torch import nn

from tinyinfer.utils.context import get_context


class PagedKVCache(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        shape = (
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        self.storage = torch.empty(shape, dtype=dtype, device=device)

    def layer_kv(self, layer_idx: int):
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]
```

现在先不写 Triton kernel。用纯 PyTorch 写入，确保地址模型正确：

```python
def store_kv(cache_k, cache_v, k, v, slot_mapping, block_size):
    # k/v: [num_tokens, num_kv_heads, head_dim]
    for i, slot in enumerate(slot_mapping.tolist()):
        block = slot // block_size
        offset = slot % block_size
        cache_k[block, offset].copy_(k[i])
        cache_v[block, offset].copy_(v[i])
```

这很慢，但非常适合验证。

Day04 的原则：

```text
先正确理解 slot → physical block/offset
再优化成 Triton kernel
```

---

# 20. Step 7：实现教学版 Prefill Attention

先用 PyTorch SDPA，不直接上 flash-attn：

```python
class Attention(nn.Module):
    def __init__(self, layer_idx, num_heads, num_kv_heads, head_dim, scale, kv_cache, block_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.kv_cache = kv_cache
        self.block_size = block_size

    def forward(self, q, k, v):
        ctx = get_context()
        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx)

        store_kv(cache_k, cache_v, k, v, ctx.slot_mapping, self.block_size)

        if ctx.is_prefill:
            return self._prefill(q, k, v, ctx)
        return self._decode(q, cache_k, cache_v, ctx)
```

为了教学清晰，prefill 先逐 sequence 切片：

```python
def _prefill(self, q, k, v, ctx):
    outputs = []
    cu_q = ctx.cu_seqlens_q.tolist()

    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        qi = q[qs:qe]  # [Tq, Hq, D]
        ki = k[qs:qe]
        vi = v[qs:qe]

        # [1, H, T, D]
        qi = qi.transpose(0, 1).unsqueeze(0)
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)

        # 教学版只处理无 cached-prefix 的最基本情况；
        # cached-prefix + new suffix 在下一小节扩展。
        oi = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=True, scale=self.scale
        )
        outputs.append(oi.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)
```

这时你已经能看到：

```text
attention kernel
```

并不是 Scheduler 负责的。Scheduler 只是构造 state；ModelRunner/Context/Attention 才消费 state。

---

# 21. Step 8：实现教学版 paged Decode Attention

Decode 的关键：根据 `block_tables` 把 historical KV gather 出来。

先写：

```python
def gather_sequence_kv(cache, block_table, context_len, block_size):
    chunks = []
    remaining = context_len

    for block_id in block_table.tolist():
        if block_id < 0 or remaining <= 0:
            break
        take = min(block_size, remaining)
        chunks.append(cache[block_id, :take])
        remaining -= take

    return torch.cat(chunks, dim=0)
```

再写 decode：

```python
def _decode(self, q, cache_k, cache_v, ctx):
    outputs = []

    for i in range(q.shape[0]):
        k_hist = gather_sequence_kv(
            cache_k,
            ctx.block_tables[i],
            int(ctx.context_lens[i].item()),
            self.block_size,
        )
        v_hist = gather_sequence_kv(
            cache_v,
            ctx.block_tables[i],
            int(ctx.context_lens[i].item()),
            self.block_size,
        )

        qi = q[i:i+1].transpose(0, 1).unsqueeze(0)
        ki = k_hist.transpose(0, 1).unsqueeze(0)
        vi = v_hist.transpose(0, 1).unsqueeze(0)

        oi = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=False, scale=self.scale
        )
        outputs.append(oi.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)
```

为什么 decode `is_causal=False` 也可以？

因为 query 只有当前最后一个 token，而 key/value 已经只包含它允许看到的历史 `[0..current]`；不存在“未来 key”。

---

# 22. Step 9：处理 GQA/MQA —— Q head 数和 KV head 数为什么可以不同？

Qwen3 可能使用：

```text
num_attention_heads = Hq
num_key_value_heads = Hkv
Hq > Hkv
```

这叫 Grouped Query Attention。

每个 K/V head 会服务多个 Q heads：

```text
queries_per_kv = Hq / Hkv
```

教学版在进入 SDPA 前可以：

```python
if self.num_kv_heads != self.num_heads:
    repeat = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
```

真正高性能实现不会喜欢这种 materialize repeat，但它帮助你先建立数学语义。

---

# 23. Step 10：把 ModelRunner 的 `run()` 接起来

`model_runner.py`：

```python
def run(self, seqs, is_prefill: bool):
    try:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        logits = self.model.compute_logits(input_ids, positions)
        temperatures = torch.tensor(
            [seq.sampling_params.temperature for seq in seqs],
            dtype=torch.float32,
            device=logits.device,
        )
        return self.model.sample(logits, temperatures).tolist()
    finally:
        reset_context()
```

这里 `compute_logits/sample` 还没实现；Day05 会用完整 Qwen3 和 Sampler 替换。

但调用边界已经全部就位：

```text
Scheduler
    ↓ seqs
ModelRunner.prepare_*
    ↓ input_ids/positions/context
Qwen3 forward
    ↓ hidden
LM head
    ↓ logits
Sampler
    ↓ token ids
Scheduler.postprocess
```

---

# 24. Step 11：为什么要先测试 metadata，而不是立刻跑模型？

创建一个 fake seq：

```python
Sequence.block_size = 4
seq = Sequence([10,11,12,13,14], SamplingParams())
seq.block_table = [7,2]
seq.num_cached_tokens = 0
```

prefill slot mapping 应为：

```text
pos0 -> 7*4+0 = 28
pos1 -> 29
pos2 -> 30
pos3 -> 31
pos4 -> 2*4+0 = 8
```

如果这里错一个数字，后面模型输出可能看起来“只是有点怪”，但实际上 KV 已经写坏了。

因此增加：

```python
def test_prefill_slot_mapping():
    # 建议把 prepare metadata 的纯逻辑拆成 CPU helper，避免测试依赖 GPU
    ...
```

原则：

```text
runtime metadata 的正确性
必须独立于 model correctness 测试
```

---

# 25. Step 12：从正确版升级到接近 nano-vLLM 的高性能版

当纯 PyTorch 路径跑通后，再做两个替换：

### 替换 A：KV write

```text
Python for-loop copy_
    ↓
Triton kernel
```

核心 kernel contract：

```text
input: k, v, slot_mapping
for token i:
    slot = slot_mapping[i]
    cache[slot] = k/v[i]
```

### 替换 B：Attention

```text
prefill:
PyTorch SDPA
    ↓
flash_attn_varlen_func


decode:
manual gather + SDPA
    ↓
flash_attn_with_kvcache / paged-kv capable kernel
```

重要的是：**优化 backend 可以替换，但 Context/slot/block-table 的 runtime contract 不变。**

---

# 26. Step 13：Prefix Cache + Prefill 的更完整语义

当：

```text
num_cached_tokens > 0
```

本轮只送 suffix token：

```text
prompt:
[0 ... 511] [512 ... 699]

cached:
[0 ... 511]

new query:
[512 ... 699]
```

但这些 query 的 attention key/value 应该是：

```text
cached prefix KV + current suffix KV
```

所以高性能 prefill kernel需要同时知道：

```text
q lengths
full context lengths
block tables / cached K/V
```

这也是为什么 prefix cache 并不是“tokenizer 层跳过前缀”这么简单；它最终改变 attention backend 如何读取历史 KV。

---

# 27. Day04 小创新：给 metadata 做“可视化 dry-run”

新增：

```text
examples/day04_metadata_dump.py
```

让它不运行模型，只打印：

```text
seq id
prompt len
cached tokens
scheduled tokens
block table
input ids range
positions
slot mapping
context len
```

例如：

```text
seq=12
cached=256
new=44
blocks=[9,2]
positions=[256..299]
slots=[512..555]  # 示例
```

这个工具后面调 Day07 Scheduler 极其有用。

建议把 dump 做成：

```bash
TINYINFER_DUMP_META=1
```

开关，而不是长期打印。

---

# 28. Day04 结束后的目录树

```text
tinyInfer/
├── examples/
│   ├── day01_toy_generate.py
│   ├── day02_scheduler_trace.py
│   └── day04_metadata_dump.py
├── tests/
│   ├── test_block_manager.py
│   ├── test_attention_meta.py
│   ├── test_scheduler.py
│   └── test_sequence.py
└── tinyinfer/
    ├── config.py
    ├── sampling_params.py
    ├── engine/
    │   ├── block_manager.py
    │   ├── llm_engine.py
    │   ├── model_runner.py
    │   ├── scheduler.py
    │   └── sequence.py
    ├── layers/
    │   └── attention.py
    └── utils/
        └── context.py
```

建议 commit：

```bash
git add .
git commit -m "day04: compile sequence state into paged-attention metadata"
```

---

# 29. 本篇验收：不看代码回答

你应该能解释：

```text
Q1. 为什么 Paged KV 不是简单把大 tensor 切小？
Q2. logical block、physical block、slot 有什么区别？
Q3. prefix cache 为什么优先复用完整 block？
Q4. ref_count 为什么必须存在？
Q5. prefill 与 decode 的 metadata 为什么不同？
Q6. cu_seqlens 解决什么问题？
Q7. block_tables 在 decode attention 中怎么被使用？
Q8. slot_mapping 和 block_table 的关系是什么？
Q9. prefix-cache hit 后为什么仍然需要 position 从 cached_tokens 开始？
Q10. 为什么先写慢而正确的 PyTorch attention，再替换 FlashAttention/Triton？
```

下一篇我们会把最后一层补全：**真正手搭 Qwen3 layers、weight loader、Sampler、Tensor Parallel、GPU warmup、KV capacity、CUDA Graph、benchmark**。完成后，tinyInfer 就不再是“toy control plane”，而是一套结构完整的 nano-vLLM 式 inference runtime。
