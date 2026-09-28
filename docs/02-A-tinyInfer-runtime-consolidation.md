# tinyInfer 补充讲义 02-A：从“逻辑可跑”到“运行时语义闭环”——仓库体检、Persistent Prefix Cache、Mixed Batching、Chunked Prefill 与系统测试

> 适用位置：完成 `01-tinyInfer-control-plane-day01-02.md` 与 `02-tinyInfer-kv-modelrunner-day03-04.md` 之后，进入真正 Qwen3 执行面之前。
>
> 本讲义不是简单增加几个 feature，而是专门解决一个很常见的学习断层：**控制面已经搭起来了，但状态语义、KV 生命周期、混合调度、Prefix Cache 和测试还没有真正闭环。**
>
> 完成本讲义后，再进入新的 03，你应该已经拥有一个“即使模型仍是 Toy/教学版，也能从调度到 paged KV metadata 完整自洽”的 runtime shell。

---

# 0. 这份 02-A 要解决什么问题？

你当前仓库已经做对了很多关键分层：

```text
LLM / LLMEngine
    ↓
Scheduler
    ↓
Sequence
    ↓
BlockManager
    ↓
ModelRunner metadata
    ↓
PagedKVCache / Attention（教学实现）
```

但现在存在四类“进入真实模型前必须先补”的问题：

1. **代码层存在直接错误或语义不一致**：例如 `attention.py` 当前不能通过语法编译，`store_kv()` 与调用名不一致，GQA helper 的方法签名也不正确。
2. **Prefix Cache 只有“活跃请求共享”，没有“请求结束后继续缓存”**：这还不是真正意义上的 cache。
3. **Scheduler 仍然是 prefill-only / decode-only batch**：只要本轮收进一个 prefill，就直接返回，decode 无法共享本轮 token budget。
4. **`num_prefix_cached_tokens` 承担了太多含义**：cache hit 数、已计算 KV 进度、本轮 prefill 起点混成一个变量，一旦加入 chunked prefill 会立刻变得含糊。
5. **pytest 覆盖不足且当前仓库测试不能形成可靠回归闭环**。

因此 02-A 的目标不是“尽快跑真实 Qwen3”，而是先把下面这张图做成立：

```text
                     ┌──────────────────────┐
new request ───────→ │ prefix cache lookup  │
                     └──────────┬───────────┘
                                │
                                v
                     num_computed_tokens
                                │
                 ┌──────────────┴───────────────┐
                 │ Scheduler: one shared budget │
                 │ decode + chunked prefill     │
                 └──────────────┬───────────────┘
                                │ SchedulerOutput
                                v
                 ┌──────────────────────────────┐
                 │ ModelRunner.prepare_batch()  │
                 │ input / positions / slots    │
                 │ q_lens / context_lens / BT   │
                 └──────────────┬───────────────┘
                                │
                                v
                     unified mixed attention
                                │
                                v
                         postprocess
                    update computed progress
                    register full cache blocks
                    sample only when required
                                │
                                v
               request finishes → ref_count--
                                │
                                v
                    cached-free LRU blocks
                    (KV remains physically valid)
```

---

# 1. 先做一次当前仓库的完整体检

当前仓库主要结构是：

```text
tinyInfer/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-tinyInfer-control-plane-day01-02.md
│   ├── 02-tinyInfer-kv-modelrunner-day03-04.md
│   └── 03-tinyInfer-qwen3-runtime-day05-09.md
├── examples/
│   ├── day01_toy_generate.py
│   └── day02_scheduler_separated.py
├── tests/
│   ├── test_attention_meta.py
│   ├── test_block_manager.py
│   ├── test_sampling_params.py
│   ├── test_scheduler.py
│   └── test_sequence.py
└── tinyinfer/
    ├── __init__.py
    ├── config.py
    ├── llm.py
    ├── sampling_params.py
    ├── engine/
    │   ├── block_manager.py
    │   ├── llm_engine.py
    │   ├── model_runner.py
    │   ├── scheduler.py
    │   └── sequence.py
    ├── layers/
    │   ├── __init__.py
    │   └── attention.py
    └── utils/
        ├── __init__.py
        ├── context.py
        └── debug.py
```

这棵树本身是合理的；现在主要问题不是“目录设计错了”，而是**各层之间的状态契约还没完全稳定**。

---

# 2. Step 0：先建立可重复测试入口

## 2.1 修改 `pyproject.toml`

当前依赖里已经写了 `xxhash`，但测试工具没有显式声明。建议补充：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

然后执行：

```bash
cd ~/Learning-Inference/tinyInfer
python -m pip install --user -e '.[dev]'
python -m pytest
```

如果你不想重新 editable install，至少确保：

```bash
python -m pip install --user pytest xxhash
export PYTHONPATH=$PWD:$PYTHONPATH
python -m pytest
```

## 2.2 当前仓库应先做静态编译检查

执行：

```bash
python -m compileall -q tinyinfer
```

你当前版本会在：

```text
tinyinfer/layers/attention.py
```

报语法错误。

这一步非常值得保留，以后每次大改都建议先跑：

```bash
python -m compileall -q tinyinfer && python -m pytest
```

---

# 3. Step 1：逐项修正当前仓库里已经存在的问题

这一节先**不增加新功能**，只把现有语义修到可继续演化。

## 3.1 `attention.py`：`_repeat()` 当前有三个问题

你现在大致是：

```python
def _repeat(k, v):
    if self.num_kv_heads != self.num_heads:
        ...
    else
        return k, v
```

问题：

```text
1. 少了 self 参数
2. else 少了冒号
3. 后面通过 self._repeat(...) 调用，因此必须是实例方法
```

修改为：

```python
def _repeat(self, k, v):
    if self.num_kv_heads == self.num_heads:
        return k, v

    if self.num_heads % self.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    repeat = self.num_heads // self.num_kv_heads
    return (
        k.repeat_interleave(repeat, dim=1),
        v.repeat_interleave(repeat, dim=1),
    )
```

注意：`repeat_interleave` 是 `torch.Tensor` 自带方法，不需要你自己定义。

## 3.2 `attention.py`：函数名调用不一致

文件顶部定义的是：

```python
def store_kv(...):
    ...
```

但 `forward()` 当前调用的是：

```python
store(...)
```

直接修改成：

```python
store_kv(
    cache_k,
    cache_v,
    k,
    v,
    ctx.slot_mapping,
    self.block_size,
)
```

## 3.3 当前 `_prefill()` 其实不能正确处理 Prefix Cache

你当前 `_prefill()` 使用：

```python
qi = q[qs:qe]
ki = k[qs:qe]
vi = v[qs:qe]
```

这只在：

```text
cached prefix = 0
```

时成立。

例如：

```text
完整上下文: A B C D E F
缓存命中:   A B C D
本轮输入:           E F
```

则：

```text
Q = Q(E,F)
K/V new = K/V(E,F)
```

但是 attention 应该看到：

```text
K/V = K/V(A,B,C,D,E,F)
```

所以 Prefix Cache 不是单纯“prepare_prefill 少传几个 token”，而是会改变 attention metadata 和 mask 语义。

**我们不会在这里临时打补丁，而是在后面直接把 prefill/decode 两条教学路径升级成 unified mixed attention。**

## 3.4 `Scheduler.schedule()` 目前不是 mixed batching

现有逻辑：

```python
if scheduled:
    return scheduled, True
```

这意味着：

```text
只要本轮成功收进任何 prefill
→ 立刻返回
→ 本轮不再安排 running decode
```

这不是 vLLM 风格的 chunked-prefill mixed batch。

## 3.5 `Scheduler.postprocess()` 对 KV 进度的假设过强

当前：

```python
self.block_manager.cache_computed_full_blocks(seq)
seq.append_token(token_id)
```

而 `cache_computed_full_blocks()` 内部直接把：

```python
seq.num_tokens
```

当作“已经完成 forward 的 token 数”。

这只有在：

```text
每个 prefill 一次性算完整 prompt
```

时勉强成立。

加入 chunked prefill 后：

```text
prompt length = 100
本轮只 schedule 32 token
```

此时：

```text
seq.num_tokens = 100
实际计算完成 = 32
```

所以必须单独维护：

```text
num_computed_tokens
```

## 3.6 `num_prefix_cached_tokens` 当前混合了三种不同概念

当前代码把它同时当作：

```text
A. Prefix Cache 命中多少 token
B. 已经具有可用 KV 的 token 数
C. 下一轮 prefill 的 input 起点
```

在“一次性 prefill”里这三个数经常相同，所以问题被掩盖了。

但 chunked prefill 后：

```text
prompt = 100
prefix hit = 32
第一轮额外 compute = 16
```

此时：

```text
num_cached_tokens   = 32
num_computed_tokens = 48
next prefill start  = 48
```

它们已经不同。

这就是下一步必须重构 `Sequence` 的原因。

## 3.7 当前 Prefix Cache 在请求结束后被彻底清空

你已经在注释里准确意识到了这一点。

现在：

```python
block.ref_count -= 1
if block.ref_count == 0:
    hash_to_block_id.pop(...)
    block.reset()
    free_ids.append(block_id)
```

语义实际是：

```text
“共享中的 block”
而不是
“可跨请求生命周期保留的 cache”
```

真正 cache 应该是：

```text
ref_count == 0
≠
内容无效
```

它只表示：

```text
当前没有活跃 Sequence 引用它
```

但它的 KV 和 hash metadata 可以继续保留，直到未来内存压力真正需要复用这个物理 block 时才 eviction。

---

# 4. Step 2：重构 `Sequence` —— 明确三种 token 进度

修改：

```text
tinyinfer/engine/sequence.py
```

把当前：

```python
self.num_prefix_cached_tokens = 0
self.num_scheduled_tokens = 0
self.is_prefill = True
```

升级为：

```python
self.num_cached_tokens = 0
self.num_computed_tokens = 0
self.num_scheduled_tokens = 0
```

其中：

```text
num_cached_tokens
    只做统计：本请求 admission 时从 Prefix Cache 复用了多少 token

num_computed_tokens
    运行时真状态：从位置 0 开始，有多少 token 的 KV 已经可用
    它既包括 cache hit，也包括本请求实际 forward 算出来的部分

num_scheduled_tokens
    仅代表“这一轮”要新增计算多少 token
```

## 4.1 增加几个派生属性

加入：

```python
@property
def prompt_computed(self) -> bool:
    return self.num_computed_tokens >= self.num_prompt_tokens

@property
def num_prompt_tokens_remaining(self) -> int:
    return max(0, self.num_prompt_tokens - self.num_computed_tokens)

@property
def num_uncomputed_tokens(self) -> int:
    return self.num_tokens - self.num_computed_tokens

@property
def needs_decode(self) -> bool:
    return self.prompt_computed and self.num_computed_tokens < self.num_tokens
```

为什么 decode 时会出现：

```text
num_computed_tokens < num_tokens
```

例如 prompt 已经算完：

```text
[A B C]  KV 已完成
```

模型采样得到：

```text
D
```

Sequence 变成：

```text
[A B C D]
```

但 D 的 K/V **还没算**，所以：

```text
num_tokens = 4
num_computed_tokens = 3
```

下一轮 decode 正是在计算 D 的 forward，并用 D 的 logits 采样 E。

这是一个非常重要的心智模型。

## 4.2 修改 `mark_scheduled()`

```python
def mark_scheduled(self, n: int):
    if n <= 0:
        raise ValueError("scheduled token count must be positive")
    if self.num_computed_tokens + n > self.num_tokens:
        raise ValueError("cannot schedule beyond available token ids")
    self.num_scheduled_tokens = n
```

---

# 5. Step 3：先给 Sequence 写回归测试

修改：

```text
tests/test_sequence.py
```

加入：

```python
def test_computed_progress_is_not_cache_hit_count():
    seq = Sequence(
        [10, 11, 12, 13, 14, 15],
        SamplingParams(max_tokens=2),
    )

    seq.num_cached_tokens = 2
    seq.num_computed_tokens = 4

    assert seq.num_cached_tokens == 2
    assert seq.num_computed_tokens == 4
    assert seq.num_prompt_tokens_remaining == 2
```

再加：

```python
def test_sampled_token_has_no_kv_until_next_forward():
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    seq.num_computed_tokens = 3

    seq.append_token(13)

    assert seq.num_tokens == 4
    assert seq.num_computed_tokens == 3
    assert seq.needs_decode
```

执行：

```bash
python -m pytest tests/test_sequence.py
```

不要继续，直到这两个测试通过。

---

# 6. Step 4：把 Prefix Cache 升级成真正 persistent cache

现在进入本讲义第一个核心功能。

修改：

```text
tinyinfer/engine/block_manager.py
```

我们希望一个 physical block 具有三种生命周期：

```text
ACTIVE
    ref_count > 0
    正被一个或多个请求引用

CACHED_FREE
    ref_count == 0
    hash/token metadata 仍然有效
    KV tensor 仍然有效
    可以被未来请求重新命中

EMPTY/REUSED
    即将分配给完全不同内容
    老 hash 被 eviction
```

注意：我们不一定真的创建 Enum；先用状态组合表达即可。

---

# 7. Step 5：用 LRU free queue 表达“可复用但仍缓存”

当前 `deque free_ids` 的含义是：

```text
free == 内容无效
```

现在改成：

```text
free == 当前无人引用，可以被 allocator 抢走
```

这个 free block 里面**可能仍然保存有效 cache**。

为了教学清晰，建议先使用标准库：

```python
from collections import OrderedDict
```

在 `BlockManager.__init__()` 中：

```python
self.free_lru: OrderedDict[int, None] = OrderedDict(
    (i, None) for i in range(num_blocks)
)

self.hash_to_block_ids: dict[int, set[int]] = {}
```

这里为什么从：

```python
hash -> block_id
```

改成：

```python
hash -> set[block_id]
```

因为未来可能出现：

```text
两个物理 block 暂时保存完全相同的 prefix 内容
```

尤其 mixed batch / same-step duplicate prefix 下，单值索引会丢信息。

---

# 8. Step 6：给 `Block` 补 parent hash

修改 `Block`：

```python
class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None
        self.parent_hash: int = 0
        self.token_ids: tuple[int, ...] = ()

    def bind(
        self,
        token_ids: list[int],
        block_hash: int,
        parent_hash: int,
    ):
        self.token_ids = tuple(token_ids)
        self.hash = block_hash
        self.parent_hash = parent_hash

    def reset(self):
        self.ref_count = 0
        self.hash = None
        self.parent_hash = 0
        self.token_ids = ()
```

这样命中时可以验证：

```text
hash 相同
parent_hash 相同
token_ids 相同
```

教学版已经足够稳健。

---

# 9. Step 7：实现真正的“evict on reuse”

加入两个 helper：

```python
def _remove_cache_index(self, block: Block):
    if block.hash is None:
        return

    ids = self.hash_to_block_ids.get(block.hash)
    if ids is None:
        return

    ids.discard(block.block_id)
    if not ids:
        self.hash_to_block_ids.pop(block.hash, None)


def _evict(self, block: Block):
    if block.ref_count != 0:
        raise RuntimeError("cannot evict active block")

    self._remove_cache_index(block)
    block.reset()
```

然后重写 `_take_free_block()`：

```python
def _take_free_block(self) -> Block:
    if not self.free_lru:
        raise RuntimeError("KV cache exhausted")

    block_id, _ = self.free_lru.popitem(last=False)
    block = self.blocks[block_id]

    if block.ref_count != 0:
        raise RuntimeError("free-LRU corruption")

    # 如果它原来是 persistent cached block，直到此刻才真正 eviction。
    if block.hash is not None:
        self._evict(block)

    block.ref_count = 1
    return block
```

这一刻你真正实现了：

```text
请求结束
≠ KV 立即消失

真正需要复用 physical page
→ 才 eviction 老 cache
```

---

# 10. Step 8：释放请求时不要 reset cache

重写 `_release_block()`：

```python
def _release_block(self, block_id: int):
    block = self.blocks[block_id]

    if block.ref_count <= 0:
        raise RuntimeError("block refcount underflow")

    block.ref_count -= 1

    if block.ref_count == 0:
        # 内容仍然有效，只是进入可被 allocator 复用的 LRU 队列。
        self.free_lru[block_id] = None
```

然后修改 `free(seq)`：

```python
def free(self, seq: Sequence):
    # 逆序释放：tail block 通常复用价值更低，应该更早进入 LRU 老端。
    for block_id in reversed(seq.block_table):
        self._release_block(block_id)

    seq.block_table.clear()
```

为什么逆序？

```text
prefix block 越靠前
→ 越有可能被未来请求共享

tail block 越靠后
→ 越 request-specific
```

所以希望 tail 更早成为 eviction 候选。

---

# 11. Step 9：Prefix Cache 命中时，从 free-LRU 中“复活” block

增加：

```python
def _acquire_cached_block(self, block_id: int):
    block = self.blocks[block_id]

    if block.ref_count == 0:
        if block_id not in self.free_lru:
            raise RuntimeError("cached-free block missing from free LRU")
        self.free_lru.pop(block_id)

    block.ref_count += 1
```

这样：

```text
命中一个 ACTIVE cache block
→ ref_count + 1

命中一个 CACHED_FREE block
→ 从 free_lru 移除
→ ref_count 0 → 1
```

这就是 persistent cache 的核心状态转换。

---

# 12. Step 10：修正全 prompt 命中时仍需至少一次 forward 的问题

这是 Prefix Cache 很容易漏掉的一个细节。

假设：

```text
prompt = 8 tokens
block_size = 4
两个 full block 全部命中 cache
```

理论上 K/V 全有了。

但是你仍然需要：

```text
最后一个 prompt token 的 hidden state / logits
```

才能采样第一个 output token。

**KV Cache 并不保存 LM head 所需的最后 hidden state。**

因此不能让：

```text
num_scheduled_tokens = 0
```

建议在 `find_cached_prefix()` 增加：

```python
def find_cached_prefix(
    self,
    seq: Sequence,
    max_cache_hit_tokens: int | None = None,
):
    ...
```

计算 full block 上限：

```python
if max_cache_hit_tokens is None:
    max_cache_hit_tokens = seq.num_prompt_tokens

max_full_blocks = max_cache_hit_tokens // self.block_size
full_blocks = min(
    seq.num_prompt_tokens // self.block_size,
    max_full_blocks,
)
```

admission 时调用：

```python
cached_ids, cached_tokens, prev_hash = self.find_cached_prefix(
    seq,
    max_cache_hit_tokens=seq.num_prompt_tokens - 1,
)
```

于是如果 prompt 恰好 block-aligned，最后一个 block 会重新计算。

这与当前 vLLM 的核心语义一致：**即使所有 prompt KV 都能命中，也必须保留至少一个需要 forward 的 token/块来得到 logits。**

---

# 13. Step 11：重写 `find_cached_prefix()` 的候选选择

把单值：

```python
block_id = self.hash_to_block_id.get(block_hash)
```

替换为：

```python
candidate_ids = self.hash_to_block_ids.get(block_hash, set())
matched_id = None

for block_id in candidate_ids:
    block = self.blocks[block_id]
    if (
        block.parent_hash == prev_hash
        and block.token_ids == tuple(token_ids)
    ):
        matched_id = block_id
        break

if matched_id is None:
    break
```

然后：

```python
cached_ids.append(matched_id)
num_cached_tokens += self.block_size
prev_hash = block_hash
```

注意：`find_cached_prefix()` 仍然必须是**只读查询**。

真正改变 refcount，要等确定整次 allocation 能成功之后再做。

---

# 14. Step 12：重写 `try_allocate_with_prefix_cache()`

目标语义：

```text
Phase 1: lookup，不改状态
Phase 2: 检查剩余 physical blocks 是否足够
Phase 3: acquire cached blocks
Phase 4: allocate missing blocks
Phase 5: 初始化 seq computed/cache state
```

核心版本：

```python
def try_allocate_with_prefix_cache(self, seq: Sequence) -> bool:
    if seq.block_table:
        raise RuntimeError("sequence already allocated")

    cached_ids, cached_tokens, prev_hash = self.find_cached_prefix(
        seq,
        max_cache_hit_tokens=seq.num_prompt_tokens - 1,
    )

    need = seq.num_blocks - len(cached_ids)
    if need > self.num_free_blocks:
        return False

    for block_id in cached_ids:
        self._acquire_cached_block(block_id)

    seq.block_table.extend(cached_ids)

    for _ in range(need):
        block = self._take_free_block()
        seq.block_table.append(block.block_id)

    seq.num_cached_tokens = cached_tokens
    seq.num_computed_tokens = cached_tokens
    seq.last_block_hash = prev_hash
    return True
```

同时：

```python
@property
def num_free_blocks(self):
    return len(self.free_lru)
```

---

# 15. Step 13：重写 full block 注册逻辑

不要再让：

```python
cache_computed_full_blocks(seq)
```

内部猜“到底计算到哪”。

明确传入：

```python
def cache_computed_full_blocks(
    self,
    seq: Sequence,
    upto_token: int,
):
```

完整实现思路：

```python
def cache_computed_full_blocks(self, seq: Sequence, upto_token: int):
    full_blocks = upto_token // self.block_size

    prev_hash = 0
    for logical_idx in range(full_blocks):
        block_id = seq.block_table[logical_idx]
        block = self.blocks[block_id]
        token_ids = seq.block_token_ids(logical_idx)

        block_hash = self._hash_block(prev_hash, token_ids)

        # 已经是相同 cache identity，则跳过重复注册。
        if (
            block.hash == block_hash
            and block.parent_hash == prev_hash
            and block.token_ids == tuple(token_ids)
        ):
            prev_hash = block_hash
            continue

        # 当前物理块理论上在这一刻第一次“内容稳定”。
        # 如果此前有其他 cache identity，先清旧索引。
        self._remove_cache_index(block)
        block.bind(token_ids, block_hash, prev_hash)
        self.hash_to_block_ids.setdefault(block_hash, set()).add(block_id)
        prev_hash = block_hash

    seq.last_block_hash = prev_hash
```

这里的关键变化是：

```text
“cache hit 数”
不再被这个函数偷偷修改。
```

`num_cached_tokens` 是统计；`num_computed_tokens` 是 runtime progress；cache index 注册是 allocator metadata。三者彻底分开。

---

# 16. Step 14：Persistent Prefix Cache 最小 pytest

修改：

```text
tests/test_block_manager.py
```

## 16.1 请求结束后 cache 仍保留

```python
def test_finished_request_keeps_cached_block():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1, 2, 3, 4, 9], SamplingParams())
    assert mgr.try_allocate_with_prefix_cache(a)
    a.num_computed_tokens = a.num_prompt_tokens
    mgr.cache_computed_full_blocks(a, a.num_computed_tokens)

    cached_block = a.block_table[0]
    mgr.free(a)

    block = mgr.blocks[cached_block]
    assert block.ref_count == 0
    assert block.hash is not None
    assert block.token_ids == (1, 2, 3, 4)
```

## 16.2 新请求可以命中已经结束请求留下的 KV

```python
def test_reuse_cache_after_original_request_finished():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=8, block_size=4)

    a = Sequence([1, 2, 3, 4, 9], SamplingParams())
    assert mgr.try_allocate_with_prefix_cache(a)
    a.num_computed_tokens = a.num_prompt_tokens
    mgr.cache_computed_full_blocks(a, a.num_computed_tokens)
    first = a.block_table[0]
    mgr.free(a)

    b = Sequence([1, 2, 3, 4, 8], SamplingParams())
    assert mgr.try_allocate_with_prefix_cache(b)

    assert b.num_cached_tokens == 4
    assert b.block_table[0] == first
    assert mgr.blocks[first].ref_count == 1
```

## 16.3 真正内存压力出现时才 eviction

```python
def test_lru_evicts_cached_free_block_only_when_reused():
    Sequence.block_size = 4
    mgr = BlockManager(num_blocks=2, block_size=4)

    a = Sequence([1, 2, 3, 4], SamplingParams())
    assert mgr.try_allocate_with_prefix_cache(a)
    a.num_computed_tokens = 4
    mgr.cache_computed_full_blocks(a, 4)
    old_hash = mgr.blocks[a.block_table[0]].hash
    mgr.free(a)

    assert old_hash in mgr.hash_to_block_ids

    # 分配其他内容，最终需要真正拿走 cached-free block。
    c = Sequence([9, 9, 9, 9, 8], SamplingParams())
    assert mgr.try_allocate_with_prefix_cache(c)

    # 至少被重新分配的旧块必须已不再保留旧 cache identity。
    for block_id in c.block_table:
        block = mgr.blocks[block_id]
        assert not (
            block.hash == old_hash and block.token_ids == (1, 2, 3, 4)
        )
```

## 16.4 block-aligned 全命中仍留最后一块重算

```python
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
```

运行：

```bash
python -m pytest tests/test_block_manager.py
```

---

# 17. Step 15：引入 `ScheduledItem` —— mixed batching 不要再只返回 bool

当前 Scheduler 返回：

```python
(list[Sequence], is_prefill: bool)
```

一旦 batch 里同时有：

```text
seq0 decode 1 token
seq1 prefill 32 tokens
seq2 decode 1 token
```

一个全局 `is_prefill` 已经没有意义。

修改：

```text
tinyinfer/engine/scheduler.py
```

在顶部加入：

```python
from dataclasses import dataclass


@dataclass(slots=True)
class ScheduledItem:
    seq: Sequence
    start_pos: int
    num_tokens: int
    is_prefill: bool
    sample_after: bool


@dataclass(slots=True)
class SchedulerOutput:
    items: list[ScheduledItem]

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items)

    @property
    def seqs(self) -> list[Sequence]:
        return [item.seq for item in self.items]
```

`sample_after` 的含义：

```text
这一条 Sequence 在本轮 forward 完成后
是否应该从它最后一个 query 的 logits 采样新 token
```

对于：

```text
partial prefill chunk
```

它必须是：

```text
False
```

否则你会在 prompt 还没吃完时就开始生成。

---

# 18. Step 16：Mixed Batching 的调度策略先选一个明确版本

02-A 不追求 production scheduler，先实现最容易讲清楚、又接近 vLLM chunked-prefill 思想的一版：

```text
每轮共享一个 token budget

Phase A: 先安排已有 running 请求的 1-token decode
         → 保护 ITL

Phase B: 用剩余 token budget 安排 waiting prefill
         → prefill 允许 chunk

同一个 SchedulerOutput 同时包含两者
```

例如：

```text
max_num_batched_tokens = 8
running decode: 3 条
waiting prompt remaining: 20 token
```

本轮：

```text
decode = 3 × 1 = 3
prefill chunk = min(20, 8 - 3) = 5

batch:
    D0(1), D1(1), D2(1), P0(5)
```

而不是旧版：

```text
只跑 P0
或者
只跑 D0,D1,D2
```

---

# 19. Step 17：先补 decode block reservation helper

当前 `prepare_next_decode_block()` 放在 postprocess 之后提前分配。

加入 mixed/chunk 后更清晰的做法是：

```text
schedule 之前确认“本轮将计算的位置”已经有 physical slot
```

在 `BlockManager` 新增：

```python
def ensure_capacity_for_token_position(
    self,
    seq: Sequence,
    token_position: int,
) -> bool:
    required_blocks = token_position // self.block_size + 1
    missing = required_blocks - len(seq.block_table)

    if missing <= 0:
        return True
    if missing > self.num_free_blocks:
        return False

    for _ in range(missing):
        block = self._take_free_block()
        seq.block_table.append(block.block_id)
    return True
```

这样 decode 时：

```python
pos = seq.num_computed_tokens
```

只要确认：

```python
ensure_capacity_for_token_position(seq, pos)
```

就可以调度。

---

# 20. Step 18：重写 `Scheduler.schedule()` 为真正 mixed batch

建议第一版完整写成下面这种结构：

```python
def schedule(self) -> SchedulerOutput:
    items: list[ScheduledItem] = []
    token_budget = self.max_num_batched_tokens
    seq_budget = self.max_num_seqs

    # ------------------------------------------------------------
    # Phase A: decode first
    # ------------------------------------------------------------
    for seq in list(self.running):
        if token_budget <= 0 or seq_budget <= 0:
            break

        if seq.num_tokens >= self.max_model_len:
            continue

        if not seq.needs_decode:
            continue

        start = seq.num_computed_tokens
        if not self.block_manager.ensure_capacity_for_token_position(seq, start):
            continue

        seq.mark_scheduled(1)
        items.append(
            ScheduledItem(
                seq=seq,
                start_pos=start,
                num_tokens=1,
                is_prefill=False,
                sample_after=True,
            )
        )
        token_budget -= 1
        seq_budget -= 1

    # ------------------------------------------------------------
    # Phase B: chunked prefill with remaining budget
    # ------------------------------------------------------------
    waiting_count = len(self.waiting)

    for _ in range(waiting_count):
        if token_budget <= 0 or seq_budget <= 0:
            break

        seq = self.waiting[0]

        # First admission: prefix lookup + block allocation.
        if not seq.block_table:
            ok = self.block_manager.try_allocate_with_prefix_cache(seq)
            if not ok:
                break

        remaining = seq.num_prompt_tokens - seq.num_computed_tokens
        if remaining <= 0:
            raise RuntimeError("waiting prefill has no remaining prompt tokens")

        chunk = min(remaining, token_budget)
        start = seq.num_computed_tokens
        end = start + chunk

        # try_allocate_with_prefix_cache() 当前已为完整 prompt 建好 block table，
        # 因此这里只做防御检查。
        if not self.block_manager.ensure_capacity_for_token_position(seq, end - 1):
            break

        sample_after = end == seq.num_prompt_tokens
        seq.mark_scheduled(chunk)

        items.append(
            ScheduledItem(
                seq=seq,
                start_pos=start,
                num_tokens=chunk,
                is_prefill=True,
                sample_after=sample_after,
            )
        )

        token_budget -= chunk
        seq_budget -= 1

        # partial prefill 做 round-robin：放到 waiting 尾部。
        self.waiting.rotate(-1)

    return SchedulerOutput(items)
```

这里暂时有一个教学简化：

```text
首次 admission 时仍然为完整 prompt 建 block table
```

而不是严格按 chunk 逐步扩容。

为什么 02-A 可以这样？

因为当前目标是先把：

```text
调度进度
mixed metadata
persistent cache
attention 语义
```

做正确。

下一阶段再把 allocator 改成完全 incremental allocation，会更容易理解。

---

# 21. Step 19：Scheduler mixed-batch pytest

重写/扩充：

```text
tests/test_scheduler.py
```

建议先做 helper：

```python
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
```

## 21.1 同一批必须能同时出现 decode + prefill

```python
def test_mixed_batch_contains_decode_and_prefill():
    sched = Scheduler(make_config(max_num_batched_tokens=6))

    running = make_seq(4, 4)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = 4
    running.block_table = [sched.block_manager._take_free_block().block_id]
    running.append_token(99)  # 99 尚未计算 KV，因此需要 decode
    sched.running.append(running)

    waiting = make_seq(10, 2)
    sched.add(waiting)

    out = sched.schedule()

    assert any(not x.is_prefill for x in out.items)
    assert any(x.is_prefill for x in out.items)
    assert out.num_scheduled_tokens <= 6
```

更稳妥的正式测试不要直接调用 private `_take_free_block()`，可以写一个 test helper 或先让 `BlockManager.allocate()` 建表；这里先表达验收思想。

## 21.2 prefill 必须按剩余 budget chunk

```python
def test_prefill_uses_remaining_budget_after_decode():
    ...
    # 2 条 decode 消耗 2 token，总 budget=6
    # prefill 本轮最多只能拿 4 token
    assert prefill_item.num_tokens == 4
```

## 21.3 partial prefill 不应 sample

```python
assert not prefill_item.sample_after
```

## 21.4 最后一段 prompt 才 sample

构造：

```text
prompt remaining = 3
token budget 足够
```

断言：

```python
assert item.sample_after
```

---

# 22. Step 20：`Context` 必须从“batch 全局模式”升级为“每 seq metadata”

当前：

```python
class Context:
    is_prefill: bool
    ...
```

mixed batch 下这个字段不够。

修改：

```text
tinyinfer/utils/context.py
```

推荐：

```python
@dataclass(slots=True)
class Context:
    q_lens: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    max_seqlen_q: int = 0

    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    is_prefill: torch.Tensor | None = None
    sample_mask: torch.Tensor | None = None
```

其中每条 seq：

```text
q_lens[i]
    本轮这条请求新增计算多少 token

context_lens[i]
    本轮算完之后，它拥有的完整 KV 上下文长度

is_prefill[i]
    当前 item 是否仍属于 prompt ingestion

sample_mask[i]
    当前 item forward 完成后是否采样
```

`cu_seqlens_k` 在教学版 unified paged attention 中暂时可以删掉，因为 K/V 不再按 batch 拼成 dense tensor，而是按 `block_table + context_len` 从 paged cache gather。

---

# 23. Step 21：把 `prepare_prefill()` / `prepare_decode()` 合成 `prepare_batch()`

修改：

```text
tinyinfer/engine/model_runner.py
```

保留：

```python
def token_to_slot(...):
    ...
```

删除或暂时不用：

```text
prepare_prefill()
prepare_decode()
```

新增：

```python
def prepare_batch(self, output: SchedulerOutput):
    input_ids = []
    positions = []
    slot_mapping = []

    q_lens = []
    context_lens = []
    is_prefill = []
    sample_mask = []
    block_tables = []

    max_blocks = max(
        len(item.seq.block_table)
        for item in output.items
    )

    for item in output.items:
        seq = item.seq
        start = item.start_pos
        end = start + item.num_tokens

        tokens = seq.token_ids[start:end]
        pos = list(range(start, end))

        if len(tokens) != item.num_tokens:
            raise RuntimeError("scheduler/model-runner token range mismatch")

        input_ids.extend(tokens)
        positions.extend(pos)
        slot_mapping.extend(token_to_slot(seq, p) for p in pos)

        q_lens.append(item.num_tokens)
        context_lens.append(end)
        is_prefill.append(item.is_prefill)
        sample_mask.append(item.sample_after)

        table = list(seq.block_table)
        table.extend([-1] * (max_blocks - len(table)))
        block_tables.append(table)

    cu_q = [0]
    for q_len in q_lens:
        cu_q.append(cu_q[-1] + q_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
    positions = torch.tensor(positions, dtype=torch.long, device=self.device)

    set_context(
        q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),
        context_lens=torch.tensor(
            context_lens, dtype=torch.int32, device=self.device
        ),
        cu_seqlens_q=torch.tensor(
            cu_q, dtype=torch.int32, device=self.device
        ),
        max_seqlen_q=max(q_lens, default=0),
        slot_mapping=torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        ),
        block_tables=torch.tensor(
            block_tables, dtype=torch.int32, device=self.device
        ),
        is_prefill=torch.tensor(
            is_prefill, dtype=torch.bool, device=self.device
        ),
        sample_mask=torch.tensor(
            sample_mask, dtype=torch.bool, device=self.device
        ),
    )

    return input_ids, positions
```

现在 metadata 已经可以表达：

```text
item0: decode q_len=1, context_len=17
item1: prefill q_len=5, context_len=37
item2: decode q_len=1, context_len=9
```

而不需要 batch 全局 `is_prefill`。

---

# 24. Step 22：为什么 mixed attention 最容易在 causal mask 上写错？

考虑一条 prefix-hit prefill：

```text
已有 KV: positions 0..3
本轮 query: positions 4..5
完整 K/V: positions 0..5
```

此时：

```text
q_len = 2
k_len = 6
```

如果简单调用：

```python
scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
)
```

你必须非常小心它在 `L != S` 时的 causal 对齐语义。

教学版最安全的做法是**显式构造绝对位置 mask**：

```text
query pos 4 可以看 key 0..4
query pos 5 可以看 key 0..5
```

mask：

```text
       K0 K1 K2 K3 K4 K5
Q4      1  1  1  1  1  0
Q5      1  1  1  1  1  1
```

这一步正是 02-A 比原 02 更重要的地方。

---

# 25. Step 23：把 Attention 改成 unified mixed path

修改：

```text
tinyinfer/layers/attention.py
```

保留：

```text
PagedKVCache
store_kv
gather_sequence_kv
_repeat
```

删除教学分支：

```text
_prefill()
_decode()
```

新增：

```python
def _attend_one(
    self,
    q_i,
    k_hist,
    v_hist,
    query_start_pos: int,
):
    # q_i:    [Tq, Hq, D]
    # k_hist: [Tk, Hkv, D]

    q = q_i.transpose(0, 1).unsqueeze(0)
    k = k_hist.transpose(0, 1).unsqueeze(0)
    v = v_hist.transpose(0, 1).unsqueeze(0)

    k, v = self._repeat(k, v)

    tq = q_i.shape[0]
    tk = k_hist.shape[0]

    q_pos = torch.arange(
        query_start_pos,
        query_start_pos + tq,
        device=q.device,
    )
    k_pos = torch.arange(tk, device=q.device)

    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    causal = causal.unsqueeze(0).unsqueeze(0)

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal,
        is_causal=False,
        scale=self.scale,
    )
    return out.squeeze(0).transpose(0, 1)
```

然后统一 `forward()`：

```python
def forward(self, q, k, v):
    ctx = get_context()
    cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx)

    # 先把本轮所有新 K/V 写入 paged cache。
    store_kv(
        cache_k,
        cache_v,
        k,
        v,
        ctx.slot_mapping,
        self.block_size,
    )

    cu_q = ctx.cu_seqlens_q.tolist()
    outputs = []

    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        q_i = q[qs:qe]

        context_len = int(ctx.context_lens[i].item())
        q_len = qe - qs
        query_start = context_len - q_len

        k_hist = gather_sequence_kv(
            cache_k,
            ctx.block_tables[i],
            context_len,
            self.block_size,
        )
        v_hist = gather_sequence_kv(
            cache_v,
            ctx.block_tables[i],
            context_len,
            self.block_size,
        )

        outputs.append(
            self._attend_one(
                q_i,
                k_hist,
                v_hist,
                query_start_pos=query_start,
            )
        )

    return torch.cat(outputs, dim=0)
```

现在 prefill/decode 的区别只剩：

```text
q_len 是 1 还是 >1
以及是否属于 prompt 阶段
```

Attention 本身不需要两套完全独立的数学语义。

这正是后面接真实 Qwen3 时很重要的一步。

---

# 26. Step 24：为 mixed metadata 写 CPU pytest

修改：

```text
tests/test_attention_meta.py
```

建议 `ModelRunner(device="cpu")`，不要求 GPU。

伪造两个 item：

```text
seq0 decode:
    token_ids=[10,11,12,13]
    num_computed=3
    schedule pos=3, q_len=1

seq1 partial prefill:
    token_ids=[20,21,22,23,24,25]
    num_computed=2
    schedule pos=2..4, q_len=3
```

断言：

```python
assert input_ids.tolist() == [13, 22, 23, 24]
assert positions.tolist() == [3, 2, 3, 4]

ctx = get_context()
assert ctx.q_lens.tolist() == [1, 3]
assert ctx.context_lens.tolist() == [4, 5]
assert ctx.cu_seqlens_q.tolist() == [0, 1, 4]
assert ctx.is_prefill.tolist() == [False, True]
```

并确认 `slot_mapping` 正确对应两条 seq 各自 block table。

测试后记得：

```python
reset_context()
```

---

# 27. Step 25：Attention correctness 最小测试

新建：

```text
tests/test_attention.py
```

先只测：

```text
1 layer
1 q head
1 kv head
head_dim = 2
block_size = 2
CPU float32
```

目标不是性能，而是验证：

```text
prefix cached + suffix query
```

是否能看到正确历史。

测试结构：

```python
def test_prefill_suffix_attends_cached_prefix():
    ...
```

构造：

```text
cache 先写入 token0/token1 的 K/V
本轮 q/k/v 只传 token2/token3
context_len = 4
q_len = 2
query_start = 2
```

然后用一个简单 dense reference：

```python
ref = torch.nn.functional.scaled_dot_product_attention(
    q_dense,
    k_dense,
    v_dense,
    attn_mask=explicit_mask,
    is_causal=False,
    scale=scale,
)
```

断言：

```python
torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
```

再补：

```python
def test_decode_attends_full_history():
    ...
```

这两个测试通过后，你的教学版 PagedAttention 才算真的从语义上闭环。

---

# 28. Step 26：ModelRunner 的采样不能再“一条 seq 必定一个 token”

chunked prefill 之后：

```text
partial prefill item
→ 不应该 sample

final prefill item
→ sample

decode item
→ sample
```

因此建议 `ModelRunner.run()` 返回：

```python
dict[int, int]
```

即：

```text
seq_id -> sampled token id
```

而不是与输入 seq list 一一对应的 `list[int]`。

真实模型版逻辑：

```python
def run(self, output: SchedulerOutput):
    try:
        input_ids, positions = self.prepare_batch(output)
        hidden_or_logits = self.model(...)

        # 每条 item 在 flat token batch 中的最后一个 query index
        last_indices = []
        cursor = 0
        sample_items = []

        for item in output.items:
            cursor += item.num_tokens
            if item.sample_after:
                last_indices.append(cursor - 1)
                sample_items.append(item)

        if not sample_items:
            return {}

        logits = self.model.compute_logits_for_indices(
            hidden_or_logits,
            last_indices,
        )
        ...

        return {
            item.seq.seq_id: token
            for item, token in zip(sample_items, sampled)
        }
    finally:
        reset_context()
```

真实 Qwen3 接入后我们会把接口进一步整理。

---

# 29. Step 27：ToyModelRunner 也要模拟 partial prefill

修改：

```text
tinyinfer/engine/model_runner.py
```

Toy 版本：

```python
class ToyModelRunner:
    def run(self, output: SchedulerOutput):
        result = {}
        for item in output.items:
            if not item.sample_after:
                continue
            seq = item.seq
            result[seq.seq_id] = (seq.last_token + 1) % 1000
        return result
```

注意：Toy runner 不真正计算 logits，所以只是测试控制面。

---

# 30. Step 28：重写 `Scheduler.postprocess()`

这是整个 mixed runtime 的关键闭环。

建议：

```python
def postprocess(
    self,
    output: SchedulerOutput,
    sampled_tokens: dict[int, int],
):
    finished = []
    prefill_completed = []

    for item in output.items:
        seq = item.seq

        # 1. 本轮 scheduled token 的 KV 已经完成。
        seq.num_computed_tokens += item.num_tokens
        seq.num_scheduled_tokens = 0

        # 2. 到新的 computed frontier 为止，注册 newly-full blocks。
        self.block_manager.cache_computed_full_blocks(
            seq,
            upto_token=seq.num_computed_tokens,
        )

        # 3. partial prefill 不采样，直接结束本 item。
        if not item.sample_after:
            continue

        token_id = sampled_tokens[seq.seq_id]
        seq.append_token(token_id)

        # 4. 采样后检查停止条件。
        if seq.should_stop(self.eos_token_id, self.max_model_len):
            seq.status = SequenceStatus.FINISHED
            finished.append(seq)
            continue

        # 5. 如果刚完成 prompt，则进入 running decode 集合。
        if item.is_prefill:
            prefill_completed.append(seq)
            seq.status = SequenceStatus.RUNNING

    # waiting 中删除完成 prefill 或 finished 的请求。
    remove_ids = {
        seq.seq_id
        for seq in prefill_completed + finished
    }
    if remove_ids:
        self.waiting = deque(
            seq for seq in self.waiting
            if seq.seq_id not in remove_ids
        )

    # 新完成 prefill 的 seq 开始进入 running decode。
    for seq in prefill_completed:
        self.running.append(seq)

    # running 中删除 finished。
    if finished:
        done_ids = {seq.seq_id for seq in finished}
        self.running = deque(
            seq for seq in self.running
            if seq.seq_id not in done_ids
        )

    # 最后释放 request ownership；persistent cached block 不会被 reset。
    for seq in finished:
        self.block_manager.free(seq)
```

注意一个细节：

```text
final prefill 本轮计算 prompt 最后 token
→ 产生第一个 output token
→ 这个新 output token 尚无 KV
→ 下一轮自然成为 decode 的 input
```

此时：

```text
num_computed_tokens == num_prompt_tokens
num_tokens == num_prompt_tokens + 1
```

完全符合 `needs_decode`。

---

# 31. Step 29：更新 `LLMEngine.step()`

修改：

```text
tinyinfer/engine/llm_engine.py
```

从：

```python
seqs, is_prefill = self.scheduler.schedule()
...
next_tokens = self.model_runner.run(seqs, is_prefill)
self.scheduler.postprocess(seqs, next_tokens)
```

改成：

```python
def step(self):
    output = self.scheduler.schedule()
    if not output.items:
        return []

    sampled_tokens = self.model_runner.run(output)
    self.scheduler.postprocess(output, sampled_tokens)
    return output.items
```

这一步以后，整个 engine 不再依赖“一个 batch 只有一种 phase”。

---

# 32. Step 30：写第一个完整 mixed end-to-end toy test

新建：

```text
tests/test_engine.py
```

测试目标：

```text
两个短请求先进入 decode
执行过程中再有一个较长 prompt
确认 scheduler 可以 mixed batch
最终所有请求都结束
```

更简单的 deterministic 版本可以直接提前把三个 seq 加进去，然后通过小 token budget 强迫分多轮执行。

至少断言：

```python
assert scheduler.is_finished()
assert all(seq.is_finished for seq in seqs)
```

以及在 trace 中至少捕获过一次：

```text
prefill item + decode item 同批
```

可以在 `Scheduler` 增加只读 debug helper：

```python
def describe_output(output):
    return [
        {
            "seq_id": x.seq.seq_id,
            "phase": "prefill" if x.is_prefill else "decode",
            "start": x.start_pos,
            "n": x.num_tokens,
            "sample": x.sample_after,
        }
        for x in output.items
    ]
```

---

# 33. Step 31：补 `BlockManager.check_consistency()`

Persistent cache 后旧 checker 已不成立：

旧逻辑假设：

```text
free block → ref_count == 0 且无 cache
not free → ref_count > 0
```

现在应检查：

```python
def check_consistency(self):
    free_set = set(self.free_lru.keys())

    for block in self.blocks:
        if block.ref_count == 0:
            assert block.block_id in free_set
        else:
            assert block.block_id not in free_set

        if block.hash is None:
            assert block.token_ids == ()
        else:
            ids = self.hash_to_block_ids.get(block.hash, set())
            assert block.block_id in ids

    for block_hash, ids in self.hash_to_block_ids.items():
        for block_id in ids:
            block = self.blocks[block_id]
            assert block.hash == block_hash
            # 注意：这里 ref_count 可以是 0！
            # 这恰恰代表 persistent cached-free block。
```

这个 checker 非常适合在：

```bash
TINYINFER_CHECK_KV=1
```

下每轮调用。

---

# 34. Step 32：增加 Prefix Cache 统计，帮助你真正看懂 cache 行为

这是 02-A 额外推荐加入的小功能。

在 `BlockManager` 增加：

```python
self.cache_hits = 0
self.cache_misses = 0
self.evictions = 0
```

命中 full block：

```python
self.cache_hits += 1
```

查找停止：

```python
self.cache_misses += 1
```

`_evict()`：

```python
self.evictions += 1
```

加：

```python
def stats(self):
    return {
        "free_blocks": self.num_free_blocks,
        "cached_blocks": sum(
            1 for block in self.blocks if block.hash is not None
        ),
        "active_blocks": sum(
            1 for block in self.blocks if block.ref_count > 0
        ),
        "cache_hits": self.cache_hits,
        "cache_misses": self.cache_misses,
        "evictions": self.evictions,
    }
```

这样以后做 benchmark 时，不会只看到 latency，却不知道为什么变快/变慢。

---

# 35. Step 33：加入 Scheduler invariant checker

在 `Scheduler` 增加：

```python
def check_consistency(self):
    waiting_ids = {s.seq_id for s in self.waiting}
    running_ids = {s.seq_id for s in self.running}

    assert waiting_ids.isdisjoint(running_ids)

    for seq in self.waiting:
        assert seq.status is SequenceStatus.WAITING
        assert seq.num_computed_tokens <= seq.num_prompt_tokens

    for seq in self.running:
        assert seq.status is SequenceStatus.RUNNING
        assert seq.prompt_computed
        assert seq.num_computed_tokens <= seq.num_tokens
```

partial prefill 时仍是 WAITING，所以别忘了 schedule admission 时不要过早把它设成 RUNNING。

---

# 36. Step 34：统一测试目录

完成 02-A 后建议测试树扩展为：

```text
tests/
├── conftest.py
├── test_sampling_params.py
├── test_sequence.py
├── test_block_manager.py
├── test_scheduler.py
├── test_model_runner_meta.py
├── test_attention.py
└── test_engine.py
```

`conftest.py` 可以放：

```python
import pytest

from tinyinfer import SamplingParams
from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence


@pytest.fixture
def small_config():
    return Config(
        max_num_batched_tokens=8,
        max_num_seqs=4,
        max_model_len=64,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
    )


def make_seq(prompt_len: int, max_tokens: int = 4):
    return Sequence(
        list(range(prompt_len)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )
```

这样不会在每个测试文件反复手写配置。

---

# 37. Step 35：推荐的 pytest 分层执行顺序

不要每改一处就只跑全量；更高效的顺序是：

```bash
# Layer 1: pure state
python -m pytest tests/test_sequence.py

# Layer 2: allocator
python -m pytest tests/test_block_manager.py

# Layer 3: scheduler
python -m pytest tests/test_scheduler.py

# Layer 4: metadata
python -m pytest tests/test_model_runner_meta.py

# Layer 5: attention math
python -m pytest tests/test_attention.py

# Layer 6: engine integration
python -m pytest tests/test_engine.py

# Final
python -m compileall -q tinyinfer
python -m pytest
```

这套层次以后接真实模型仍然适用。

---

# 38. Step 36：加一个 deterministic debug trace

现在 `trace_enabled()` 已有雏形。

建议 Scheduler 每轮输出：

```text
[step 12]
budget=8
items=[
  seq=0 decode start=21 n=1 sample=1,
  seq=1 decode start=13 n=1 sample=1,
  seq=4 prefill start=16 n=6 sample=0,
]
kv={free=28 cached=11 active=7 evictions=2}
```

不要直接把 print 分散在各个函数里。

新增：

```text
tinyinfer/utils/debug.py
```

helper：

```python
def trace_schedule(step_id, output, block_manager):
    if not trace_enabled("TINYINFER_TRACE_SCHED"):
        return
    ...
```

这样进入 03 以后你调真实 Qwen3 错误时，可以先判断：

```text
是 model math 错了
还是调度/metadata 本来就错了
```

---

# 39. Step 37：这一阶段先不做哪些事？

为了避免 02-A 失控，下面暂时不加入：

```text
1. GPU swap / CPU offload
2. production preemption/recompute
3. multimodal cache key
4. distributed KV transfer
5. CUDA Graph
6. Triton paged-attention kernel
7. Radix tree prefix cache
8. speculative decoding
```

这些都可以后续做，但不是进入真实模型前的必要条件。

---

# 40. 02-A 结束后的推荐目录树

```text
tinyInfer/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-tinyInfer-control-plane-day01-02.md
│   ├── 02-tinyInfer-kv-modelrunner-day03-04.md
│   ├── 02-A-tinyInfer-runtime-consolidation.md
│   └── 03-tinyInfer-qwen3-runtime-day05-06-revised.md
├── examples/
│   ├── day01_toy_generate.py
│   ├── day02_scheduler_separated.py
│   └── mixed_batch_trace.py              # 推荐新增
├── tests/
│   ├── conftest.py
│   ├── test_sampling_params.py
│   ├── test_sequence.py
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   ├── test_model_runner_meta.py
│   ├── test_attention.py
│   └── test_engine.py
└── tinyinfer/
    ├── __init__.py
    ├── config.py
    ├── llm.py
    ├── sampling_params.py
    ├── engine/
    │   ├── block_manager.py
    │   ├── llm_engine.py
    │   ├── model_runner.py
    │   ├── scheduler.py
    │   └── sequence.py
    ├── layers/
    │   ├── __init__.py
    │   └── attention.py
    └── utils/
        ├── __init__.py
        ├── context.py
        └── debug.py
```

注意：我们刻意**没有在 02-A 新增大量 model/layer 文件**。这一篇只把 runtime shell 做正确。

---

# 41. 02-A 最终验收场景 1：Persistent Prefix Cache

手动画：

```text
block_size=4
num_blocks=4

Request A:
[1 2 3 4 | 5 6]

prefill 后：
physical 0 = [1 2 3 4], cached hash H0, ref=1
physical 1 = [5 6 .. ..], partial, ref=1

A finished:
physical 0: hash H0, ref=0, still cached
physical 1: ref=0, uncached/free

Request B:
[1 2 3 4 | 9 9]

lookup:
logical block0 → reuse physical0
ref: 0→1
只需计算 [9 9]
```

如果你仍然认为：

```text
A finished → physical0 内容被 reset
```

则 persistent cache 尚未实现。

---

# 42. 最终验收场景 2：Mixed Batching

配置：

```text
max_num_batched_tokens = 8
```

已有：

```text
seq0 decode 1
seq1 decode 1
```

新请求：

```text
seq2 prompt remaining = 20
```

本轮应能得到：

```text
seq0 decode: 1
seq1 decode: 1
seq2 prefill: 6
----------------
total = 8
```

而不是：

```text
只 decode 2
```

也不是：

```text
只 prefill 8
```

---

# 43. 最终验收场景 3：Chunked Prefill

```text
prompt length = 20
budget = 8
```

无其他请求时：

```text
step0: positions 0..7,  sample=False
step1: positions 8..15, sample=False
step2: positions 16..19, sample=True → 生成第一个 output
step3: decode output token
```

你必须能解释为什么：

```text
只有 step2 才 sample
```

---

# 44. 最终验收场景 4：Prefix Cache + Chunked Prefill + Mixed Batch 同时出现

这是进入 03 前最重要的一题。

假设：

```text
block_size = 4
budget = 8

seq0: decode 1 token
seq1 prompt = [A B C D | E F G H | I J K L]
      前 4 token 命中 cache
```

则：

```text
seq0 先占 1 token budget
seq1 num_computed_tokens 初始 = 4
剩余 budget = 7

seq1 本轮 schedule positions 4..10，共 7 token
context_len = 11
q_len = 7

attention 的每个 query 必须能看到：
Q(position 4) → K 0..4
...
Q(position10) → K 0..10
```

本轮结束：

```text
seq1 num_computed_tokens = 11
prompt 尚余 1 token
sample=False
```

下一轮继续。

如果这道题你能完全画出：

```text
block_table
slot_mapping
q_lens
context_lens
causal mask
```

那么 02-A 的目的就真正达到了。

---

# 45. 02-A 结束时你必须能回答的 12 个问题

1. `num_cached_tokens` 和 `num_computed_tokens` 为什么不能共用一个字段？
2. 为什么新 sample 出来的 token 还不能计入 `num_computed_tokens`？
3. 为什么 Prefix Cache 只缓存 full block 更简单？
4. 为什么 request finished 不等于 block 内容立即失效？
5. `ref_count==0` 为什么可以仍然是一个有效 cache block？
6. eviction 应该发生在 `free()` 时还是 physical block 真正被重新分配时？
7. 为什么 free block 本身也可以有 hash？
8. 为什么全 prompt 命中 cache 时仍需要至少一次 forward？
9. mixed batching 后为什么不能再用一个全局 `is_prefill`？
10. partial prefill 为什么不能 sample？
11. prefix-hit prefill 为什么不能直接用普通 `is_causal=True` 而不检查 q/k 对齐？
12. Scheduler、BlockManager、ModelRunner、Attention 各自应该拥有哪些状态，哪些状态绝对不应该互相偷管？

---

# 46. 推荐 commit 划分

不要一次 commit 全部功能。

推荐：

```bash
git add .
git commit -m "test: establish runtime regression suite"

git add .
git commit -m "refactor: separate cached and computed token progress"

git add .
git commit -m "feat: keep prefix cache blocks after request completion"

git add .
git commit -m "feat: add LRU eviction for cached free KV blocks"

git add .
git commit -m "feat: add chunked prefill mixed batching"

git add .
git commit -m "refactor: unify mixed-batch model metadata"

git add .
git commit -m "fix: support cached-prefix attention with explicit causal mask"

git add .
git commit -m "test: cover mixed scheduling and paged attention semantics"
```

这样以后你回顾 git history 时，整个 runtime 的演化会非常清楚。

---

# 47. 下一篇 03 的新起点

新的 03 不再假设：

```text
一个 batch = 全 prefill 或全 decode
```

而是直接建立在：

```text
SchedulerOutput(items)
    ↓
ModelRunner.prepare_batch()
    ↓
flat token tensor
+ q_lens
+ context_lens
+ block_tables
+ slot_mapping
+ sample_mask
    ↓
unified paged attention
```

之上。

换句话说，03 的任务终于可以单纯聚焦：

```text
把 ToyModel / fake projection
逐层替换为真实 Qwen3
```

而不用一边写模型，一边继续修 runtime 基础语义。
