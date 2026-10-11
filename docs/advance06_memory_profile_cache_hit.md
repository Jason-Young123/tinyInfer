# tinyInfer Advance 06：Prefill Memory Profiling 与 Prefix Cache Hit Rate

本讲以 `tinyInfer-master(20261011-063957).zip` 为唯一源码基线，沿用 `docs/advance04_flashattention.md` / `docs/advance05_flashdecoding.md` 的逐步教学结构。

## 本讲目标

- GPU Resource Statistics = **Reserved + Weights + Temporary Peak + Allocated For KV + Available For KV**。
- `seq_num=1`、`tokens=max_num_batched_tokens`，用全部来自 `hello` 的输入做一次模型 backbone dummy Prefill，调用 `torch.cuda.max_memory_allocated()`。
- 小型临时 Paged KV Cache 的占用包含在 Temporary Peak 内，不单独拆分；不使用 Engine/Scheduler 进行 profile。
- 每个用户 Last request 的 E2E 下显示单次 Cache Hit Rate；右侧 TPOT percentiles 下显示全局平均命中率。
- 左右两列独立滚动；右侧包括 Statistics 和 Controller 的整列滚动。
- 下列七份代码均为**完整文件内容**。相对于上传基线的新行/改动行尾均加 `# [MOD]`；JavaScript 使用 `// # [MOD]`，HTML 使用 `<!-- # [MOD] -->`，CSS 使用 `/* # [MOD] */`，以保持源码可运行。未改动的旧标记原样保留。

## 0. 核心概念与预算

```text
Total = GPU device total_memory
Utilization = config.gpu_memory_utilization
Reserved = Total × (1 - Utilization)
Engine Budget = Total - Reserved
Weights = sum(parameter.numel × parameter.element_size)
Profile Peak = torch.cuda.max_memory_allocated() after one dummy prefill
Temporary Peak = max(0, Profile Peak - Weights)
KV Budget = Engine Budget - Weights - Temporary Peak
Num Blocks = floor(KV Budget / bytes_per_kv_block)
```

**边界说明：** `max_memory_allocated()` 计算 PyTorch allocator 已分配张量峰值，不等于整个进程的总显存峰值，也不一定完全覆盖外部 CUDA workspace 或非 PyTorch 的显存占用；该预估适合本项目简化教学目标，不应当视为严格的生产级显存保证。`Temporary Peak` 还会包含参数之外的常驻张量及 dummy KV，因而命名为保守的非权重峰值预算。

本讲中 UI 的 KV 两项表示**逻辑预算分布**，而 `PagedKVCache.storage` 在初始化时是**整池预分配**：

```text
Allocated For KV = active_block_count × block_bytes
Available For KV = Total - Reserved - Weights - Temporary Peak - Allocated For KV
```

Available For KV 包括空闲、cached-free 物理 KV blocks 的预算，以及无法凑成完整 block 的尾部余量。因此该图不是实时 `torch.cuda.memory_allocated()` 分解图。五段严格相加等于 Total，且 allocated 不超过整个 pool。

命中率按 admission 数据计算：

```text
Single Request Cache Hit Rate = seq.num_cached_tokens / seq.num_prompt_tokens
Global Cache Hit Rate = sum(completed request cache hit rates) / completed requests
```

举例：请求 A 的命中率 50%，请求 B 为 0%，全局为 25%，与两条请求 token 数是否相同无关。

## 1. 修改范围及顺序

```text
tinyinfer/
├── engine/
│   ├── model_runner.py    # dummy prefill + KV 显存预算
│   ├── llm_engine.py      # 五段 Resource snapshot
│   └── sequence.py        # 单请求 Cache Hit Rate
├── runtime/
│   └── dynamic_engine.py  # 全局平均 Cache Hit Rate
└── ui/static/
    ├── index.html         # Resource + Performance 静态标签
    ├── app.js             # User panel 和全局渲染
    └── style.css          # 右侧独立滚动
```

启动流程为：加载 Qwen3 权重 → 按 hello token 生成一次 dummy input → 临时 KV + 一次 backbone forward → 读取 peak → 销毁临时 KV → 计算 blocks → 正式创建共享 KV pool → 启动 scheduler。请按 Step 1 到 Step 7 替换文件。

## Step 1：独立模型 Dummy Prefill 与 KV block 预算

**修改文件：** `tinyinfer/engine/model_runner.py`

将原先基于 `memory_reserved()` 的估算换成模型 backbone 前向峰值。`AutoTokenizer` 直接把 hello 文本变成 token IDs，补齐到 `max_num_batched_tokens`，不经过 `LLMEngine`、`Scheduler` 或 `Sequence`。因为本项目 Attention 强制通过 `Context` 查询 Paged KV，需临时分配足够容纳这一条 Prefill 的小 KV，并设置一次 Context；完整 forward 后释放，临时 KV 本身计入峰值。

**完整修改后文件：**

```python
from dataclasses import dataclass
import torch

from tinyinfer.utils.context import set_context, reset_context
from tinyinfer.engine.sequence import SchedulerOutput
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights


# 将seq内的逻辑token position映射到paged KV的flat slot
def token_to_slot(seq, token_position: int) -> int:
    block_size = seq.block_size
    logical_block = token_position // block_size
    offset = token_position % block_size
    if logical_block >= len(seq.block_table):
        raise RuntimeError("token position has no allocated physical block")
    physical_block = seq.block_table[logical_block]
    return physical_block * block_size + offset


def dtype_nbytes(dtype:torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

# 计算一个token在所有层中所需要的KV cache容量大小
def bytes_per_kv_block(hf_config, block_size: int, dtype: torch.dtype) -> int:
    head_dim = int(
        getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
    )
    return (
        int(hf_config.num_hidden_layers)
        * 2  # K + V
        * int(block_size)
        * int(hf_config.num_key_value_heads)
        * head_dim
        * dtype_nbytes(dtype)
    )


# 显存静态分区：Reserved + Weights + Temporary Peak + KV budget = Total # [MOD]
@dataclass(frozen=True, slots=True)
class KVMemoryProfile:
    total_bytes: int
    budget_bytes: int
    reserved_bytes: int
    weights_bytes: int # [MOD]
    temporary_peak_bytes: int # [MOD]
    block_bytes: int
    num_blocks: int


def build_kv_memory_profile(config, device, dtype, model) -> KVMemoryProfile: # [MOD]
    if device.type != "cuda":
        raise ValueError("Invalid device: cpu; Required: cuda")
    from transformers import AutoTokenizer # [MOD]

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = int(total_bytes * config.gpu_memory_utilization)
    reserved_bytes = total_bytes - budget_bytes # [MOD]
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) # [MOD]
    block_bytes = bytes_per_kv_block(config.hf_config, config.kvcache_block_size, dtype)
    if block_bytes <= 0:
        raise RuntimeError("invalid KV bytes per block")

    # 直接把 hello 文本编码为指定长度，不经过 Engine/Scheduler # [MOD]
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True) # [MOD]
    hello_ids = tokenizer.encode("hello", add_special_tokens=False) # [MOD]
    if not hello_ids: # [MOD]
        raise RuntimeError("hello tokenizer returned no tokens") # [MOD]
    n = config.max_num_batched_tokens # [MOD]
    ids = (hello_ids * ((n + len(hello_ids) - 1) // len(hello_ids)))[:n] # [MOD]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device) # [MOD]
    positions = torch.arange(n, dtype=torch.long, device=device) # [MOD]
    num_dummy_blocks = (n + config.kvcache_block_size - 1) // config.kvcache_block_size # [MOD]
    head_dim = int(getattr(config.hf_config, "head_dim", config.hf_config.hidden_size // config.hf_config.num_attention_heads)) # [MOD]
    dummy_cache = PagedKVCache( # [MOD]
        num_layers=config.hf_config.num_hidden_layers, # [MOD]
        num_blocks=num_dummy_blocks, # [MOD]
        block_size=config.kvcache_block_size, # [MOD]
        num_kv_heads=config.hf_config.num_key_value_heads, # [MOD]
        head_dim=head_dim, dtype=dtype, device=device, # [MOD]
    ) # [MOD]
    model.set_kv_cache(dummy_cache) # [MOD]
    try: # [MOD]
        torch.cuda.reset_peak_memory_stats(device) # [MOD]
        with torch.inference_mode(): # [MOD]
            set_context( # [MOD]
                cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=device), # [MOD]
                context_lens=torch.tensor([n], dtype=torch.int32, device=device), # [MOD]
                block_tables=torch.arange(num_dummy_blocks, dtype=torch.int32, device=device).unsqueeze(0), # [MOD]
                slot_mapping=torch.arange(n, dtype=torch.long, device=device), # [MOD]
            ) # [MOD]
            model(input_ids, positions, output_hidden_states=False) # [MOD]
        torch.cuda.synchronize(device) # [MOD]
        profile_peak_bytes = int(torch.cuda.max_memory_allocated(device)) # [MOD]
    finally: # [MOD]
        reset_context() # [MOD]
        for layer in model.model.layers: # [MOD]
            layer.self_attn.attn.kv_cache = None # [MOD]
        del dummy_cache, input_ids, positions # [MOD]
        torch.cuda.empty_cache() # [MOD]

    temporary_peak_bytes = max(0, profile_peak_bytes - weights_bytes) # [MOD]
    kv_budget_bytes = budget_bytes - weights_bytes - temporary_peak_bytes # [MOD]
    num_blocks = kv_budget_bytes // block_bytes
    if num_blocks <= 0:
        raise RuntimeError("no memory left for KV cache after dummy prefill profiling") # [MOD]
    return KVMemoryProfile(
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        reserved_bytes=reserved_bytes,
        weights_bytes=weights_bytes, # [MOD]
        temporary_peak_bytes=temporary_peak_bytes, # [MOD]
        block_bytes=int(block_bytes),
        num_blocks=int(num_blocks),
    )



class ToyModelRunner:
    def run1(self, sequences, is_prefill: bool): # 当前阶段不论是不是prefill都默认在末尾添加 id+1
        next_tokens = []
        for seq in sequences:
            next_tokens.append((seq.last_token + 1) % 1000)
        return next_tokens

    def run(self, output: SchedulerOutput) -> dict[int, int]:
        result = {}
        for item in output.items:
            if item.sample_after:
                result[item.seq.seq_id] = (item.seq.last_token + 1) % 1000
        return result


# Real model runner
class ModelRunner:
    def __init__(self, config, device:str="cuda", dtype:torch.dtype=torch.bfloat16): # 把model从初始化列表中移除, 因为model固定为Qwen3ForCausalLM
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.sampler = Sampler()
        
        if self.config.hf_config is None:
            self.config.load_hf_config()
        hf_config = self.config.hf_config
        hf_config.kvcache_block_size = self.config.kvcache_block_size

        # step1: 先不绑定kv cache, 单纯构造模型
        self.model = Qwen3ForCausalLM(hf_config, kv_cache = None).to(device = self.device, dtype = self.dtype)

        # step2: 加载真实的Qwen3 checkpoint
        report = load_weights(self.model, self.config.model, strict = True)
        self.load_report = report
        self.model.eval()

        # step3: 模型参数驻留后再估算KV容量
        #num_blocks = estimate_num_kv_blocks(self.config, self.device, self.dtype)
        #self.config.num_kvcache_blocks = num_blocks
        self.resource_profile = build_kv_memory_profile(self.config, self.device, self.dtype, self.model) # [MOD]
        num_blocks = self.resource_profile.num_blocks
        self.config.num_kvcache_blocks = num_blocks
        head_dim = int(
            getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
        )

        # step4: 创建唯一共享的PagedKVCache
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.config.kvcache_block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # step5: 绑定到所有 decoder layers, 对应到所有attn
        self.model.set_kv_cache(self.kv_cache)



    # 对多个seq准备prefill workload; deprecated
    # 假设一个batch包含3个prefill请求, seq0 = [A, B, C, D, | E, F], seq1 = [a, b, | c, d], seq2 = [|1, 2, 3]; (|右侧代表尚未进入prefix cache的部分)
    # 则执行完prepare_prefill之后:
    #  input_ids = [E, F, c, d, 1, 2, 3]
    #  positions = [4, 5, 2, 3, 0, 1, 2]
    #  slot_mapping = [81, 82, 23, 24, 56, 57, 58]
    #  q_lens = [2, 2, 3]; k_lens = [6, 4, 3]
    #  cu_q = [0, 2, 4, 7]; cu_k = [0, 6, 10, 13]
    def prepare_prefill(self, seqs): 
        input_ids = [] # 真正送给模型计算的 token, 而非完整prompt; 多个seq请求直接拼接而成
        positions = [] # 这些 token 在原始 sequence 中的位置; 多个seq请求直接拼接而成
        slot_mapping = [] # 注意这里仅代表本批prefill最终写入KV cache的slot, 不包含读取的prefix cache对应的slot
        q_lens = []    # 每个seq本次prefill需要真正计算的长度
        k_lens = []    # 每个seq本次attention看到的历史token长度

        for seq in seqs:
            start = seq.num_prefix_cached_tokens # 已经位于prefix cache中的token数目
            end = seq.num_tokens
            tokens = seq.token_ids[start:end]
            pos = list(range(start, end)) # 真正需要prefill的token下标范围

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

        cu_q = prefix_sum(q_lens) # 导出flashAttention所需要的cumsum形式
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

    # 对多个seq准备decode workload; deprecated
    # 假设一个batch同样包含3个decode请求, seq0 = [A, B, C], seq1 = [a, b, c, d], seq2 = [1, 2];
    # 则执行完prepare_docode之后(假设block_size = 2):
    #  input_ids = [C, d, 2]
    #  positions = [2, 3, 1]
    #  slot_mapping = [60, 71, 12]
    #  context_lens = [3, 4, 2]
    #  block_tables = [[11, 12], [34, 35], [59, -1]], 所有历史token对应的KV Cache block编号
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
            table.extend([-1] * (max_blocks - len(table))) # 末尾补-1, 代表这个seq比较短、后续block用不到
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

    # 真正的统一调度接口, 实现了mixed batching
    def prepare_batch(self, output: SchedulerOutput) -> tuple[torch.Tensor, torch.Tensor]:
        if not output.items:
            raise ValueError("cannot prepare an empty SchedulerOutput")
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        q_lens: list[int] = []
        context_lens: list[int] = []
        is_prefill: list[bool] = []
        is_sample: list[bool] = []
        block_tables: list[list[int]] = []
        max_blocks = max(len(item.seq.block_table) for item in output.items) # 注意这里的block已经被预分配

        for item in output.items: # type: ScheduledItems
            seq = item.seq
            start = item.start_pos
            end = start + item.num_tokens

            tokens = seq.token_ids[start:end] # 需要送入模型的tokens
            pos = list(range(start, end))

            if len(tokens) != item.num_tokens: # 逻辑上冗余, 但python的切片在越界时不会报错, 因此需要保留这个检测
                raise RuntimeError("scheduler/model-runner token range mismatch")
            
            input_ids.extend(tokens) # 各个seq请求的输入tokens直接拼接
            positions.extend(pos)    # 各个seq请求的tokens绝对位置直接拼接
            slot_mapping.extend(token_to_slot(seq, p) for p in pos) # 各个seq请求的tokens对应的physical slot直接拼接

            q_lens.append(item.num_tokens)      # 每个seq请求本轮输入tokens的数目
            context_lens.append(end)            # 每个seq请求本轮前向传播完毕后,总上下文长度
            is_prefill.append(item.is_prefill)  # 每个seq请求本轮是否处于prefill阶段
            is_sample.append(item.sample_after) # 每个seq请求本轮是否需要采样下一个token

            table = list(seq.block_table)       # 每个seq请求本轮涉及到的block_table, 且长度向max_len对齐, 末尾补-1
            table.extend([-1] * (max_blocks - len(table)))
            block_tables.append(table)

        cu_q = [0]
        for q_len in q_lens:
            cu_q.append(cu_q[-1] + q_len)       # 额外对q_lens求cumsum数组

        input_ids_ret = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions_ret = torch.tensor(positions, dtype=torch.long, device=self.device)

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
            is_sample=torch.tensor(
                is_sample, dtype=torch.bool, device=self.device
            ),
        )

        return input_ids_ret, positions_ret



    @torch.no_grad() # 这个函数里的计算不需要构建 autograd 计算图，也不需要反向传播; 适合纯推理任务
    def run(self, output: SchedulerOutput) -> dict[int, int]:
        try:
            input_ids, positions = self.prepare_batch(output) # shape = [num_flat_tokens]
            # 一个batch全部送入模型进行前向传播 
            hidden_states, _ = self.model(input_ids, positions, output_hidden_states = False) # shape = [num_flat_tokens, hidden_dim]

            # flat hidden_states 中只抽取需要 sample 的每条 sequence 最后一个 query
            sample_flat_indices: list[int] = []
            sample_items = []
            cursor = 0

            for item in output.items:
                cursor += item.num_tokens
                if item.sample_after:
                    sample_flat_indices.append(cursor - 1) # hidden_states中需要被采样的token向量的位置; 注意不论是last prefill还是decode, 默认一个seq请求仅有一个token需要采样
                    sample_items.append(item)
            if not sample_items: # 没有采样需求, 说明当前batch全都是partial prefill, 无final prefill或者decode的seq请求
                return {}

            idx = torch.tensor(sample_flat_indices, dtype=torch.long, device=hidden_states.device) # [sampled_seq], 即这一批中is_sample=True的seq请求总数
            sample_hidden = hidden_states.index_select(0, idx) # [sampled_seq, hidden_dim]
            logits = self.model.compute_logits(sample_hidden)  # [sampled_seq, vocab_size]

            temperatures = torch.tensor( # 为每个需要sample的seq设置temperature, shape = [sampled_seq]
                [item.seq.sampling_params.temperature for item in sample_items],
                dtype=logits.dtype,
                device=logits.device,
            )
            greedy_mask = torch.tensor( # shape = [sampled_seq]
                [item.seq.sampling_params.greedy for item in sample_items],
                dtype=torch.bool,
                device=logits.device,
            )

            tokens = self.sampler(logits, temperatures, greedy_mask)

            return { # 第几个item增加了哪个token; 并非所有item都会增加token
                item.seq.seq_id: int(token) for item, token in zip(sample_items, tokens.tolist())
            }
        finally:
            # Context 是一次 forward 的动态状态，绝不能泄漏到下一轮
            reset_context()
```

## Step 2：修改 Resource Snapshot 五段口径

**修改文件：** `tinyinfer/engine/llm_engine.py`

UI 显示总显存的五部分：Reserved、Weights、Temporary Peak、Allocated For KV、Available For KV。这里的 `Allocated For KV` 仍按 active block 的**逻辑容量**计算；缓存空闲块虽然实际已经由物理 KV pool 预分配，却归入 Available For KV。这是资源预算视图，而不是 CUDA allocator 的实时物理分账。

**完整修改后文件：**

```python
from dataclasses import dataclass
import torch

from transformers import AutoTokenizer
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


@dataclass(slots=True)
class StreamUpdate:
    seq_id: int
    token_id: int
    completion_token_ids: list[int]
    finished: bool
    statistics: dict | None


# 整体调用链条: LLM -> LLMEngine -> 创建seqs列表 -> Scheduler -> ModelRunner
class LLMEngine:
    def __init__(self, config: Config | None = None, device: str = "cuda", dtype=torch.bfloat16):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        self.config.load_hf_config()

        Sequence.block_size = self.config.kvcache_block_size

        # ModelRunner 先初始化，因为它会根据真实模型显存占用计算config.num_kvcache_blocks;
        # 随后 Scheduler/BlockManager 才能使用该值。
        self.model_runner = ModelRunner(self.config, device=device, dtype=dtype)
        self.scheduler = Scheduler(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        self.config.eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1)
        self.scheduler.eos_token_id = self.config.eos_token_id


    def add_request(self, token_ids: list[int], params: SamplingParams): # 简化后的创建请求函数, 用一个token list代表
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq) # 进入waiting_list
        return seq.seq_id
 
    def build_chat_prompt(self, text:str, system_prompt:str|None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        if self.tokenizer.chat_template:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return text

    # 把 UI 字符串转换成真正 chat-template prompt, 再进入原 Scheduler; 等价于generate函数中的把prompt转化为token id的步骤
    def add_text_request( # 创建seq的同时记录arrival_time
        self,
        text: str,
        params: SamplingParams,
        system_prompt: str | None = None,
    ) -> int:
        prompt = self.build_chat_prompt(text, system_prompt)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.add_request(token_ids, params)


    def step_with_updates(self) -> tuple[list, list[StreamUpdate]]:
        # 对内: 走通schedule -> run -> postprocess的完整一轮调度流程
        output = self.scheduler.schedule() # 从waiting池子中挑选本轮进行推理的请求
        if not output.items:
            return [], []
        sampled_tokens = self.model_runner.run(output) # 输出为dict of {seq_id: token_id}, 表示所有需要采样的请求所产生的下一个token id是什么
        self.scheduler.postprocess(output, sampled_tokens) # 将产生的下一个token拼接到对应的seq请求中, 并注册prefix cache; 管理waiting/running list

        # 对外: 把当前一轮真正生成token的请求信息打包送给前端, 进行网页端聊天框刷新
        updates = []
        for item in output.items:
            token_id = sampled_tokens.get(item.seq.seq_id) # 得到产生next token请求的next token id
            if token_id is None: # 如果这一批的某个seq没有生成下一个token则直接跳过, 无需进行前端网页刷新
                continue
            seq = item.seq
            updates.append(
                StreamUpdate(
                    seq_id=seq.seq_id,
                    token_id=token_id,
                    completion_token_ids=list(
                        seq.token_ids[seq.num_prompt_tokens:]
                    ),
                    finished=seq.is_finished,
                    statistics=seq.statistics() if seq.is_finished else None,
                )
            )

        return output.items, updates # 前者是内部信息(这一轮调度了哪些请求, 不论是否生成next token); 后者是对外信息(这一轮哪些请求生成了next token从而需要刷新聊天框)

    def step(self): # 对内的前向传播调度函数, 不对外传递信息; 仅静态调度会调用这个函数
        items, _ = self.step_with_updates()
        return items

    
    # 资源快照只读取 CPU metadata，不触发 GPU synchronize # [MOD]
    def runtime_resource_snapshot(self) -> dict:
        profile = self.model_runner.resource_profile
        blocks = self.scheduler.block_manager.blocks
        active_block_ids = [block.block_id for block in blocks if block.ref_count > 0] # [MOD]
        cached_block_ids = [block.block_id for block in blocks if block.ref_count == 0 and block.hash is not None] # [MOD]
        allocated_kv_bytes = len(active_block_ids) * profile.block_bytes # [MOD]
        kv_pool_bytes = profile.num_blocks * profile.block_bytes # [MOD]
        available_kv_bytes = profile.total_bytes - profile.reserved_bytes - profile.weights_bytes - profile.temporary_peak_bytes - allocated_kv_bytes # [MOD]
        return {
            "total_bytes": profile.total_bytes, # [MOD]
            "reserved_bytes": profile.reserved_bytes, # [MOD]
            "weights_bytes": profile.weights_bytes, # [MOD]
            "temporary_peak_bytes": profile.temporary_peak_bytes, # [MOD]
            "allocated_kv_bytes": allocated_kv_bytes, # [MOD]
            "available_kv_bytes": available_kv_bytes, # [MOD]
            "kv_pool_bytes": kv_pool_bytes, # [MOD]
            "block_bytes": profile.block_bytes, # [MOD]
            "num_blocks": len(blocks),
            "active_block_ids": active_block_ids, # [MOD]
            "cached_block_ids": cached_block_ids, # [MOD]
        }


    # 以下这两个函数均针对静态调度: 所有请求同步到达, 并在全部结束推理后返回完整output
    # 针对所有seq请求生成/decode完整的token id序列
    def generate_token_ids(
        self,
        prompts: list[list[int]], # 每个请求本身是一个list,因此这里是list的list
        sampling_params: SamplingParams | list[SamplingParams], # 如果只有一个sampling_params说明所有请求共用, 需要进行广播
    ):
        # basic argument check
        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = sampling_params
        if len(params) != len(prompts):
            raise ValueError("prompts and sampling params length mismatch")

        # 创建Sequence列表并加入到Scheduler中由其进行调度
        seqs = []
        for token_ids, p in zip(prompts, params): # seq达到一刻
            seq = Sequence(token_ids, p)
            seqs.append(seq)
            self.scheduler.add(seq)

        # 直到Scheduler调度完成, 所有请求全部推理完毕; 实际上Scheduler会调用ModelRunner进行真正的推理
        while not self.scheduler.is_finished():
            items = self.step()
            if not items and not self.scheduler.is_finished():
                raise RuntimeError("scheduler made no progress while requests remain")

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:], # decode阶段生成的token id, 不包含prompt
                "all_token_ids": list(seq.token_ids), # 所有的token_id
                "num_cached_tokens": seq.num_cached_tokens, # prefix cache命中的token数目
                "statistics": seq.statistics(),
            }
            for seq in seqs
        ]


    # 上层最终调用的接口, 输入输出均为str列表
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        # step1: str -> token_id
        prompt_token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]

        # step2: 运行直到每个seq请求都推理完毕, 得到完整的token_ids列表
        outputs = self.generate_token_ids(prompt_token_ids, sampling_params)

        # step3: 将输出的token_ids 解码为字符串
        for out in outputs:
            out["text"] = self.tokenizer.decode(out["token_ids"], skip_special_tokens=True) # 动态为dict新建"text"键

        return outputs
```

## Step 3：单请求 Cache Hit Rate

**修改文件：** `tinyinfer/engine/sequence.py`

按 `num_cached_tokens / num_prompt_tokens` 计算，不新增缓存检索。`num_cached_tokens` 在 admission 时设置，因此会反映本次请求的初始 prefix reuse。

**完整修改后文件：**

```python
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
import time

from tinyinfer.sampling_params import SamplingParams


# 计算单条请求内部 ITL 的分位数。
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


class SequenceStatus(Enum):
    WAITING = auto()   # 已进入引擎，等待 Scheduler 调度
    RUNNING = auto()   # 正在参与 prefill / decode
    FINISHED = auto()  # 已完成，不再参与调度


class Sequence:
    counter = count()   # 给每个 Sequence 分配全局递增 seq_id
    block_size = 256    # 一个 KV block 可容纳的 token 数; 注意这里的block_size为逻辑大小(token数目)而非物理大小(Byte)

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params #每个 seq 可以有不同的采样参数

        # token_ids 始终保存：prompt tokens + 已生成 tokens
        self.token_ids = list(token_ids)
        # 静态参数, 不像 num_tokens 一样是动态property参数
        self.num_prompt_tokens = len(token_ids) # 创建Sequence对象时已经确定, 后续token_ids可能在逐渐变长但num_prompts_tokens不会再改变

        # KV / 调度相关状态
        # self.num_prefix_cached_tokens = 0       # 已经存在 KV cache、无需本轮重新计算的 prefix token 数
        # self.num_scheduled_tokens = 0    # scheduler 决定本轮要真正送进模型的 token 数; 和后续 chunked prefill 配合
        # self.is_prefill = True           # True: prefill；False: decode
        self.num_cached_tokens = 0              # 本请求 admission 时从 Prefix Cache 复用了多少 token
        self.num_computed_tokens = 0            # 运行时真状态：从位置 0 开始，有多少 token 的 KV 已经可用
        self.num_scheduled_tokens = 0           # 这一轮要新增计算多少 token
        self.block_table: list[int] = []        # logical KV block -> physical block id
        self.last_block_hash: int = 0           # 最近一个满block对应的hash code

        # performance
        self.arrival_time = time.perf_counter() # 创建seq的时间戳; seq对象创建时即记录arrival_time
        self.first_scheduled_time: float | None = None # 第一次被调度的时间戳
        self.first_token_time: float | None = None # TTFT
        self.finished_time: float | None = None
        self.output_token_times: list[float] = []

    @property
    def last_token(self) -> int:
        """返回当前最后一个 token，decode 阶段通常只需要它作为新输入。"""
        return self.token_ids[-1]

    @property
    def num_tokens(self) -> int: # 有property修饰的成员函数可以直接像访问成员变量一样访问
        """当前总 token 数 = prompt + completion。"""
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        """已经生成的 token 数。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        """当前请求是否已经结束。"""
        return self.status is SequenceStatus.FINISHED

    @property
    def num_blocks(self) -> int: # 对于当前的length需要多少个物理block
        n = self.num_tokens
        return (n + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self) -> int:
        rem = self.num_tokens % self.block_size
        return rem if rem else self.block_size

    @property # prompt的prefill是否已经完成
    def prompt_computed(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property # prompt还剩多少没有完成
    def num_prompt_tokens_remaining(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property # 是否还需要继续decode, 满足两个条件: 1. prefill已经完成 + 2. 还有tokens没进KV cache
    def needs_decode(self) -> bool:
        return self.prompt_computed and self.num_computed_tokens < self.num_tokens



    # logic_block_id -> 该逻辑block内所有token
    def block_token_ids(self, logical_idx: int) -> list[int]:
        begin = logical_idx * self.block_size
        end = min(begin + self.block_size, self.num_tokens)
        return self.token_ids[begin:end]

    def mark_scheduled(self, n: int): # 把本轮要计算KV cache的token数设为n
        if n <= 0:
            raise ValueError("scheduled token count must be positive")
        if self.num_computed_tokens + n > self.num_tokens: # 比如decode阶段通常每轮针对1个token计算KV cache, 如果为2则不满足条件
            raise ValueError("cannot schedule beyond available token ids")
        self.num_scheduled_tokens = n
        if self.first_scheduled_time is None: # 首次被调度
            self.first_scheduled_time = time.perf_counter()

    def append_token(self, token_id: int):
        """把模型新生成的 token 追加到当前 Sequence。"""
        self.token_ids.append(int(token_id))

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.output_token_times.append(now) # 同步计时

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.finished_time = time.perf_counter()
        #print("only for test: ", self.finished_time)


    def should_stop(self, eos_token_id: int, max_model_len: int) -> bool:
        """达到 completion 上限、模型上下文上限，或生成 EOS 时停止。"""
        # case1: 用户指定的最大生成长度
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True

        # case2: 模型允许的最大总上下文长度：
        # prompt tokens + completion tokens
        if self.num_tokens >= max_model_len:
            return True

        # case3: 遇到 EOS
        if (
            not self.sampling_params.ignore_eos
            and eos_token_id >= 0
            and self.last_token == eos_token_id
        ):
            return True

        return False


    # 性能统计: 单位统一为s;
    def statistics(self) -> dict[str, int | float | None]:
        # token statistics
        prompt_tokens = self.num_prompt_tokens
        output_tokens = self.num_completion_tokens
        decode_tokens = max(0, output_tokens - 1)
        total_tokens = self.num_tokens
        cached_tokens = self.num_cached_tokens

        # arrival -> first scheduled
        queue_time = None
        if self.first_scheduled_time is not None:
            queue_time = self.first_scheduled_time - self.arrival_time

        # first scheduled -> first output token
        prefill_time = None
        if self.first_scheduled_time is not None and self.first_token_time is not None:
            prefill_time = self.first_token_time - self.first_scheduled_time

        # first output token -> finished
        decode_time = None
        if self.first_token_time is not None and self.finished_time is not None:
            decode_time = self.finished_time - self.first_token_time

        # arrival -> first output token
        ttft = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.arrival_time

        # first scheduled -> finished
        service_time = None
        if self.first_scheduled_time is not None and self.finished_time is not None:
            service_time = self.finished_time - self.first_scheduled_time

        # arrival -> finished
        e2e_latency = None
        if self.finished_time is not None:
            e2e_latency = self.finished_time - self.arrival_time

        # average latency per decode token
        tpot = None
        if decode_time is not None and decode_tokens > 0:
            tpot = decode_time / decode_tokens

        # adjacent output-token latency
        itls = [
            self.output_token_times[i] - self.output_token_times[i - 1]
            for i in range(1, len(self.output_token_times))
        ]
        mean_itl = sum(itls) / len(itls) if itls else None
        min_itl = min(itls) if itls else None
        max_itl = max(itls) if itls else None
        # User panel 显示上一条请求内部的 TPOT/ITL 分位数。
        tpot_p50 = _percentile(itls, 0.50)
        tpot_p95 = _percentile(itls, 0.95)
        tpot_p99 = _percentile(itls, 0.99)

        # steady-state decode throughput
        decode_throughput = None
        if decode_time is not None and decode_time > 0 and decode_tokens > 0:
            decode_throughput = decode_tokens / decode_time

        # whole-request output throughput
        request_throughput = None
        if e2e_latency is not None and e2e_latency > 0:
            request_throughput = output_tokens / e2e_latency

        return {
            # token statistics
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens, # 输出总token
            "decode_tokens": decode_tokens, # decode输出总token, = output_tokens - 1
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0, # [MOD]

            # latency
            "queue_time": queue_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "ttft": ttft,
            "service_time": service_time,
            "e2e_latency": e2e_latency,
            "tpot": tpot,
            "tpot_p50": tpot_p50,
            "tpot_p95": tpot_p95,
            "tpot_p99": tpot_p99,
            "mean_itl": mean_itl,
            "min_itl": min_itl,
            "max_itl": max_itl,

            # throughput
            "decode_throughput": decode_throughput,
            "request_throughput": request_throughput,
        }





# 最小调度单元, 一个处于chunked prefill/decode阶段的seq
@dataclass(slots=True)
class ScheduledItem:
    seq: Sequence
    start_pos: int
    num_tokens: int # 这一轮对于该seq而言需要送入模型前向传播的token数
    is_prefill: bool
    # 这一条 Sequence 在本轮 forward 完成后是否应该从它最后一个 query 的 logits 采样新 token; 对于chunked prefill阶段为false
    sample_after: bool 


# 调度单元集合, 针对一批次推理
@dataclass(slots=True)
class SchedulerOutput:
    items: list[ScheduledItem]

    @property # 这一批一共要处理多少个token
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items)

    @property # 这一批有多少个seq
    def seqs(self) -> list[Sequence]:
        return [item.seq for item in self.items]
```

## Step 4：全局已完成请求平均命中率

**修改文件：** `tinyinfer/runtime/dynamic_engine.py`

全局平均使用每条已完成请求命中率的算术平均（macro average），而不是先累加 cached tokens 再除以全部 prompt tokens。每个完成请求只汇总一次；与现有 Performance 分位数路径一致。

**完整修改后文件：**

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable

from tinyinfer.engine.llm_engine import LLMEngine
from tinyinfer.sampling_params import SamplingParams


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]



PERFORMANCE_KEYS = (
    "queue_time",
    "prefill_time",
    "decode_time",
    "ttft",
    "service_time",
    "e2e_latency",
    "tpot",
    "decode_throughput",
    "request_throughput",
)


@dataclass(slots=True)
class SubmitRequest:
    user_id: str
    text: str
    params: SamplingParams


@dataclass(slots=True)
class RuntimeEvent:
    type: str
    user_id: str | None = None
    seq_id: int | None = None
    text: str | None = None
    statistics: dict | None = None
    resource: dict | None = None
    performance: dict | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DynamicEngineService: # 将LLMEngine包装为动态调度引擎
    def __init__(
        self,
        engine: LLMEngine,
        system_prompt: str = "You are a helpful, concise assistant.",
        snapshot_interval_s: float = 0.25,
    ):
        self.engine = engine
        self.system_prompt = system_prompt
        self.snapshot_interval_s = snapshot_interval_s

        self._commands: Queue[list[SubmitRequest] | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._event_sink: Callable[[dict], None] | None = None # 最后指向

        self._busy_users: set[str] = set()
        self._busy_lock = Lock()
        self._seq_to_user: dict[int, str] = {}

        self._metric_sums = {key: 0.0 for key in PERFORMANCE_KEYS}
        self._metric_counts = {key: 0 for key in PERFORMANCE_KEYS}
        self._completed_requests = 0
        self._tpot_values: list[float] = []
        self._cache_hit_rate_sum = 0.0 # [MOD]

        self._last_snapshot_at = 0.0
        self._snapshot_lock = Lock()
        self._last_runtime_snapshot: dict | None = None

    def start(self, event_sink: Callable[[dict], None]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._event_sink = event_sink
        self._stop.clear()
        self._thread = Thread(
            target=self._worker_loop,
            name="tinyinfer-engine",
            daemon=True,
        )
        self._thread.start() # 开启_worker_loop线程

    def stop(self) -> None:
        self._stop.set()
        self._commands.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)

    def submit(self, request: SubmitRequest) -> bool:
        return bool(self.submit_many([request]))

    def submit_many(self, requests: list[SubmitRequest]) -> list[str]:
        accepted = []
        batch = []

        with self._busy_lock:
            for request in requests:
                if not request.text.strip():
                    continue
                if request.user_id in self._busy_users:
                    continue
                self._busy_users.add(request.user_id)
                accepted.append(request.user_id)
                batch.append(request)

        if batch:
            self._commands.put(batch)
        return accepted

    def is_user_busy(self, user_id: str) -> bool:
        with self._busy_lock:
            return user_id in self._busy_users

    def current_runtime_snapshot(self) -> dict | None:
        with self._snapshot_lock:
            return deepcopy(self._last_runtime_snapshot)

    def _emit(self, event: RuntimeEvent) -> None:
        if self._event_sink:
            self._event_sink(event.to_dict())

    def _release_user(self, user_id: str) -> None:
        with self._busy_lock:
            self._busy_users.discard(user_id)

    def _admit_batch(self, batch: list[SubmitRequest]) -> None:
        for request in batch:
            try:
                seq_id = self.engine.add_text_request( # 关键逻辑: 进入waiting list
                    request.text,
                    request.params,
                    self.system_prompt,
                )
                self._seq_to_user[seq_id] = request.user_id
                self._emit(
                    RuntimeEvent(
                        type="started",
                        user_id=request.user_id,
                        seq_id=seq_id,
                    )
                )
            except Exception as exc:
                self._release_user(request.user_id)
                self._emit(
                    RuntimeEvent(
                        type="error",
                        user_id=request.user_id,
                        message=str(exc),
                    )
                )

    def _drain_commands(self) -> None:
        while True:
            try:
                batch = self._commands.get_nowait()
            except Empty:
                return
            if batch is None:
                return
            self._admit_batch(batch)

    def _record_statistics(self, statistics: dict | None) -> None:
        if not statistics:
            return
        self._completed_requests += 1
        self._cache_hit_rate_sum += float(statistics.get("cache_hit_rate", 0.0)) # [MOD]
        for key in PERFORMANCE_KEYS:
            value = statistics.get(key)
            if value is None:
                continue
            numeric = float(value)
            self._metric_sums[key] += numeric
            self._metric_counts[key] += 1
            if key == "tpot":
                self._tpot_values.append(numeric)

    def _performance_snapshot(self) -> dict:
        averages = {}
        for key in PERFORMANCE_KEYS:
            count = self._metric_counts[key]
            averages[key] = (
                self._metric_sums[key] / count if count else None
            )
        return {
            "completed_requests": self._completed_requests,
            "cache_hit_rate": self._cache_hit_rate_sum / self._completed_requests if self._completed_requests else None, # [MOD]
            "averages": averages,
            # 所有已完成 seq 的 request-level TPOT 分布
            "tpot_percentiles": {
                "p50": _percentile(self._tpot_values, 0.50),
                "p95": _percentile(self._tpot_values, 0.95),
                "p99": _percentile(self._tpot_values, 0.99),
            },
        }

    def _emit_runtime_snapshot(self, force: bool = False) -> None: # 输出当前runtime的状态快照, 包括resource和perf
        now = time.perf_counter()
        if not force and now - self._last_snapshot_at < self.snapshot_interval_s:
            return
        self._last_snapshot_at = now

        snapshot = {
            "resource": self.engine.runtime_resource_snapshot(),
            "performance": self._performance_snapshot(),
        }
        with self._snapshot_lock:
            self._last_runtime_snapshot = deepcopy(snapshot)

        self._emit(
            RuntimeEvent(
                type="runtime",
                resource=snapshot["resource"],
                performance=snapshot["performance"],
            )
        )

    def _handle_updates(self, updates) -> bool:
        any_finished = False
        for update in updates:
            user_id = self._seq_to_user.get(update.seq_id)
            if user_id is None:
                continue

            text = self.engine.tokenizer.decode(
                update.completion_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self._emit(
                RuntimeEvent(
                    type="token",
                    user_id=user_id,
                    seq_id=update.seq_id,
                    text=text,
                )
            )

            if update.finished:
                any_finished = True
                self._record_statistics(update.statistics) # 记录全局的perf statistics
                self._emit(
                    RuntimeEvent(
                        type="finished",
                        user_id=user_id,
                        seq_id=update.seq_id,
                        text=text,
                        statistics=update.statistics,
                    )
                )
                self._seq_to_user.pop(update.seq_id, None)
                self._release_user(user_id)
        return any_finished

    def _fail_active_requests(self, message: str) -> None:
        for seq_id, user_id in list(self._seq_to_user.items()):
            self._emit(
                RuntimeEvent(
                    type="error",
                    user_id=user_id,
                    seq_id=seq_id,
                    message=message,
                )
            )
            self._release_user(user_id)
        self._seq_to_user.clear()

    def _worker_loop(self) -> None: # 最核心的循环函数
        self._emit_runtime_snapshot(force=True)

        while not self._stop.is_set():
            if self.engine.scheduler.is_finished():
                try:
                    batch = self._commands.get(timeout=0.1)
                except Empty:
                    self._emit_runtime_snapshot()
                    continue
                if batch is None:
                    continue
                self._admit_batch(batch)

            self._drain_commands()
            if self.engine.scheduler.is_finished():
                continue

            try:
                items, updates = self.engine.step_with_updates()
                if not items and not self.engine.scheduler.is_finished():
                    raise RuntimeError(
                        "scheduler made no progress while requests remain"
                    )
                any_finished = self._handle_updates(updates) # 检查是否有seq请求推理完毕; 刷新全局perf statistics
                self._emit_runtime_snapshot(force=any_finished)
            except Exception as exc:
                self._fail_active_requests(str(exc))
                self._emit_runtime_snapshot(force=True)
                return
```

## Step 5：HTML 五段资源和命中率 UI

**修改文件：** `tinyinfer/ui/static/index.html`

调整图例和条形图五个分区，Performance statistics 的 TPOT percentiles 下新增 Cache Hit Rate。

**完整修改后文件：**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tinyInfer Console</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="shell">
    <section class="users" id="users"></section>

    <aside class="side">
      <section class="panel statistics-panel">
        <div class="panel-head global-head">
          <div>
            <div class="panel-title">Statistics</div>
            <div class="subtle">Runtime telemetry · refresh ~4 Hz</div>
          </div>
          <div class="live-dot"><span></span>live</div>
        </div>

        <section class="stats-section resource-section">
          <div class="section-title">Resource statistics</div>

          <div class="memory-bar" id="memory-bar" aria-label="GPU memory budget composition">
            <div class="memory-segment reserved" id="segment-reserved"></div>
            <div class="memory-segment weights" id="segment-weights"></div> <!-- # [MOD] -->
            <div class="memory-segment temporary" id="segment-temporary"></div> <!-- # [MOD] -->
            <div class="memory-segment allocated" id="segment-allocated"></div>
            <div class="memory-segment available" id="segment-available"></div>
          </div>

          <div class="memory-legend">
            <div><span class="swatch reserved"></span><span>Reserved</span><strong id="mem-reserved">--</strong></div>
            <div><span class="swatch weights"></span><span>Weights</span><strong id="mem-weights">--</strong></div> <!-- # [MOD] -->
            <div><span class="swatch temporary"></span><span>Temporary Peak</span><strong id="mem-temporary">--</strong></div> <!-- # [MOD] -->
            <div><span class="swatch allocated"></span><span>Allocated For KV</span><strong id="mem-allocated">--</strong></div> <!-- # [MOD] -->
            <div><span class="swatch available"></span><span>Available For KV</span><strong id="mem-available">--</strong></div> <!-- # [MOD] -->
          </div>

          <div class="kv-head">
            <div class="kv-summary-wrap">
              <div class="section-label">KV cache blocks</div>
              <!-- [MOD] block数量与单block容量拆成两行，给右侧三态图例留足空间。 -->
              <div class="subtle" id="kv-summary">Waiting for runtime snapshot</div>
              <div class="subtle kv-block-size" id="kv-block-size">-- / block</div>
            </div>
            <!-- [MOD] 三态图例固定为单行。 -->
            <div class="kv-key">
              <span class="kv-key-item"><span class="swatch block-empty"></span>empty</span>
              <span class="kv-key-item"><span class="swatch block-cached"></span>cached</span>
              <span class="kv-key-item"><span class="swatch block-active"></span>active</span>
            </div>
          </div>
          <div class="block-grid" id="block-grid"></div>
        </section>

        <section class="stats-section performance-section">
          <div class="section-title-row">
            <div class="section-title">Performance statistics</div>
            <div class="request-count" id="completed-requests">0 completed</div>
          </div>
          <div class="performance-grid" id="performance-grid"></div>
          <!-- [MOD] 全局 TPOT 分位数：对所有已完成请求的 request-level TPOT 取分位数。 -->
          <div class="percentile-head">TPOT percentiles</div>
          <div class="percentile-grid" id="performance-tpot-percentiles">
            <div class="metric-card"><span>P50</span><strong data-tpot-percentile="p50">--</strong></div>
            <div class="metric-card"><span>P95</span><strong data-tpot-percentile="p95">--</strong></div>
            <div class="metric-card"><span>P99</span><strong data-tpot-percentile="p99">--</strong></div>
          </div>
          <div class="percentile-head">Cache Hit Rate</div> <!-- # [MOD] -->
          <div class="percentile-grid"><div class="metric-card"><span>Completed requests mean</span><strong id="performance-cache-hit-rate">--</strong></div></div> <!-- # [MOD] -->
        </section>
      </section>

      <section class="panel controller">
        <div class="panel-title">Controller</div>
        <div class="controller-grid">
          <button id="start-all" class="primary" disabled>Start all ready</button>
          <div class="controller-slot">future</div>
          <div class="controller-slot">future</div>
          <div class="controller-slot">future</div>
        </div>
      </section>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

## Step 6：前端请求指标与五段绘制

**修改文件：** `tinyinfer/ui/static/app.js`

每个 User panel 的 Last request 在 E2E 下新增 Cache Hit Rate；每次请求完成时显示本次命中率，右侧随 runtime 快照显示所有已完成请求的平均值。

**完整修改后文件：**

```javascript
// [MOD] User panel 从 3 个扩展到 5 个。
const USERS = ["user1", "user2", "user3", "user4", "user5"];
const PERFORMANCE_METRICS = [
  ["queue_time", "Queue", "latency"],
  ["prefill_time", "Prefill", "latency"],
  ["decode_time", "Decode", "latency"],
  ["ttft", "TTFT", "latency"],
  ["service_time", "Service", "latency"],
  ["e2e_latency", "E2E", "latency"],
  ["tpot", "TPOT", "latency"],
  ["decode_throughput", "Decode tok/s", "throughput"],
  ["request_throughput", "Request tok/s", "throughput"],
];

const states = new Map();
const usersRoot = document.querySelector("#users");
const startAll = document.querySelector("#start-all");
const blockGrid = document.querySelector("#block-grid");
const performanceGrid = document.querySelector("#performance-grid");
const ws = new WebSocket(`ws://${location.host}/ws`);

let blockNodes = [];
let activeBlocks = new Set();
let cachedBlocks = new Set(); // [MOD] 独立保存 cached-free block。

function makePanel(userId, index) {
  const panel = document.createElement("section");
  panel.className = "panel user-panel";
  panel.innerHTML = `
    <div class="panel-head">
      <div class="panel-title">User ${index + 1}</div>
      <div class="status idle">Idle</div>
    </div>
    <div class="output"></div>
    <div class="user-lower">
      <div class="composer">
        <textarea placeholder="Type a prompt..."></textarea>
        <button class="primary start-button" disabled>Start</button>
      </div>
      <div class="user-tools">
        <section class="mini-panel request-controls">
          <div class="mini-title">Controller</div>
          <div class="control-row">
            <span>max tokens</span>
            <input class="max-tokens" type="range" min="32" max="8192" step="32" value="8192">
            <span class="control-value max-value">8192</span>
          </div>
          <div class="control-row">
            <span>temperature</span>
            <input class="temperature" type="range" min="0.1" max="2.0" step="0.1" value="0.8">
            <span class="control-value temp-value">0.8</span>
          </div>
          <div class="toggle-row">
            <span>greedy</span>
            <label class="switch">
              <input class="greedy" type="checkbox">
              <span class="slider-toggle"></span>
            </label>
          </div>
        </section>
        <section class="mini-panel user-statistics">
          <div class="mini-title">Last request</div>
          <div class="user-stat-grid">
            <div class="user-stat-row single">
              <div class="user-stat"><span>TTFT</span><strong class="stat-ttft">--</strong></div>
            </div>
            <div class="user-stat-row pair">
              <div class="user-stat"><span>TPOT mean</span><strong class="stat-tpot">--</strong></div>
              <div class="user-stat"><span>TPOT p50</span><strong class="stat-tpot-p50">--</strong></div>
            </div>
            <div class="user-stat-row pair">
              <div class="user-stat"><span>TPOT p95</span><strong class="stat-tpot-p95">--</strong></div>
              <div class="user-stat"><span>TPOT p99</span><strong class="stat-tpot-p99">--</strong></div>
            </div>
            <div class="user-stat-row single">
              <div class="user-stat"><span>E2E</span><strong class="stat-e2e">--</strong></div>
            </div>
            <div class="user-stat-row single"> <!-- # [MOD] -->
              <div class="user-stat"><span>Cache Hit Rate</span><strong class="stat-cache-hit-rate">--</strong></div> <!-- # [MOD] -->
            </div> <!-- # [MOD] -->
          </div>
        </section>
      </div>
    </div>`;

  const state = {
    userId,
    panel,
    input: panel.querySelector("textarea"),
    button: panel.querySelector(".start-button"),
    output: panel.querySelector(".output"),
    status: panel.querySelector(".status"),
    maxTokens: panel.querySelector(".max-tokens"),
    maxValue: panel.querySelector(".max-value"),
    temperature: panel.querySelector(".temperature"),
    tempValue: panel.querySelector(".temp-value"),
    greedy: panel.querySelector(".greedy"),
    statTTFT: panel.querySelector(".stat-ttft"),
    statTPOT: panel.querySelector(".stat-tpot"),
    statTPOTP50: panel.querySelector(".stat-tpot-p50"), // [MOD]
    statTPOTP95: panel.querySelector(".stat-tpot-p95"), // [MOD]
    statTPOTP99: panel.querySelector(".stat-tpot-p99"), // [MOD]
    statE2E: panel.querySelector(".stat-e2e"),
    statCacheHitRate: panel.querySelector(".stat-cache-hit-rate"), // # [MOD]
    busy: false,
    outputBase: "",
  };
  states.set(userId, state);

  state.input.addEventListener("input", refreshControls);
  state.button.addEventListener("click", () => submitOne(state));
  state.maxTokens.addEventListener("input", () => {
    state.maxValue.textContent = state.maxTokens.value;
  });
  state.temperature.addEventListener("input", () => {
    state.tempValue.textContent = Number(state.temperature.value).toFixed(1);
  });

  usersRoot.appendChild(panel);
}

function ready(state) {
  return !state.busy && state.input.value.trim().length > 0;
}

function setBusy(state, busy) {
  state.busy = busy;
  state.input.readOnly = busy;
  state.maxTokens.disabled = busy;
  state.temperature.disabled = busy;
  state.greedy.disabled = busy;
  state.status.textContent = busy ? "Running" : "Idle";
  state.status.classList.toggle("busy", busy);
  state.status.classList.toggle("idle", !busy);
  refreshControls();
}

function refreshControls() {
  for (const state of states.values()) {
    state.button.disabled = !ready(state) || ws.readyState !== WebSocket.OPEN;
  }
  startAll.disabled =
    ws.readyState !== WebSocket.OPEN ||
    ![...states.values()].some(ready);
}

function requestPayload(state) {
  return {
    user_id: state.userId,
    text: state.input.value,
    max_tokens: Number(state.maxTokens.value),
    temperature: Number(state.temperature.value),
    greedy: state.greedy.checked,
  };
}

function beginLocal(state) {
  state.outputBase = state.output.textContent;
  setBusy(state, true);
}

function submitOne(state) {
  if (!ready(state) || ws.readyState !== WebSocket.OPEN) return;
  const payload = requestPayload(state);
  beginLocal(state);
  ws.send(JSON.stringify({type: "submit", ...payload}));
}

startAll.addEventListener("click", () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  const requests = [];
  for (const state of states.values()) {
    if (!ready(state)) continue;
    requests.push(requestPayload(state));
    beginLocal(state);
  }
  if (requests.length) {
    ws.send(JSON.stringify({type: "submit_many", requests}));
  }
});

function formatLatency(value) {
  if (value == null) return "--";
  return `${(value * 1000).toFixed(value < 0.1 ? 1 : 0)} ms`;
}

function formatSeconds(value) {
  if (value == null) return "--";
  return `${value.toFixed(2)} s`;
}

function formatThroughput(value) {
  if (value == null) return "--";
  return `${value.toFixed(2)} tok/s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const mib = bytes / (1024 ** 2);
  if (mib < 1024) return `${mib.toFixed(2)} MB`;
  return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
}

function formatMemoryPart(bytes, total) {
  if (!total) return "--";
  const percent = 100 * bytes / total;
  return `${formatBytes(bytes)} · ${percent.toFixed(1)}%`;
}

function formatRate(value) { // # [MOD]
  return value == null ? "--" : `${(value * 100).toFixed(1)}%`; // # [MOD]
} // # [MOD]

function updateUserStatistics(state, statistics) {
  if (!statistics) return;
  state.statTTFT.textContent = formatLatency(statistics.ttft);
  state.statTPOT.textContent = formatLatency(statistics.tpot);
  state.statTPOTP50.textContent = formatLatency(statistics.tpot_p50); // [MOD]
  state.statTPOTP95.textContent = formatLatency(statistics.tpot_p95); // [MOD]
  state.statTPOTP99.textContent = formatLatency(statistics.tpot_p99); // [MOD]
  state.statE2E.textContent = formatSeconds(statistics.e2e_latency);
  state.statCacheHitRate.textContent = formatRate(statistics.cache_hit_rate); // # [MOD]
}

function setSegment(id, bytes, total) {
  const node = document.querySelector(id);
  const percent = total > 0 ? Math.max(0, 100 * bytes / total) : 0;
  node.style.width = `${percent}%`;
}

function ensureBlockGrid(numBlocks) {
  if (blockNodes.length === numBlocks) return;
  blockGrid.replaceChildren();
  blockNodes = new Array(numBlocks);
  activeBlocks = new Set();
  cachedBlocks = new Set(); // [MOD]

  const fragment = document.createDocumentFragment();
  for (let blockId = 0; blockId < numBlocks; blockId += 1) {
    const node = document.createElement("div");
    node.className = "kv-block empty"; // [MOD] 默认状态明确为 empty。
    node.title = `block ${blockId} · empty`;
    blockNodes[blockId] = node;
    fragment.appendChild(node);
  }
  blockGrid.appendChild(fragment);
}

// [MOD] 一个 block 只属于 empty / cached / active 三种状态之一。
function paintBlock(blockId, activeSet, cachedSet) {
  const node = blockNodes[blockId];
  if (!node) return;

  node.classList.remove("empty", "cached", "active");
  if (activeSet.has(blockId)) {
    node.classList.add("active");
    node.title = `block ${blockId} · active`;
  } else if (cachedSet.has(blockId)) {
    node.classList.add("cached");
    node.title = `block ${blockId} · cached`;
  } else {
    node.classList.add("empty");
    node.title = `block ${blockId} · empty`;
  }
}

function updateBlockGrid(numBlocks, activeBlockIds, cachedBlockIds) {
  ensureBlockGrid(numBlocks);
  const nextActive = new Set(activeBlockIds ?? []);
  const nextCached = new Set(cachedBlockIds ?? []);

  // [MOD] 只重绘状态发生过变化的 block，避免每次刷新扫描全部 DOM。
  const touched = new Set([
    ...activeBlocks,
    ...cachedBlocks,
    ...nextActive,
    ...nextCached,
  ]);
  for (const blockId of touched) {
    paintBlock(blockId, nextActive, nextCached);
  }

  activeBlocks = nextActive;
  cachedBlocks = nextCached;
}

function updateResource(resource) {
  if (!resource) return;
  const total = resource.total_bytes ?? 0;
  const reserved = resource.reserved_bytes ?? 0;
  const weights = resource.weights_bytes ?? 0; // # [MOD]
  const temporary = resource.temporary_peak_bytes ?? 0; // # [MOD]
  const allocated = resource.allocated_kv_bytes ?? 0;
  const available = resource.available_kv_bytes ?? 0;

  document.querySelector("#mem-reserved").textContent = formatMemoryPart(reserved, total);
  document.querySelector("#mem-weights").textContent = formatMemoryPart(weights, total); // # [MOD]
  document.querySelector("#mem-temporary").textContent = formatMemoryPart(temporary, total); // # [MOD]
  document.querySelector("#mem-allocated").textContent = formatMemoryPart(allocated, total);
  document.querySelector("#mem-available").textContent = formatMemoryPart(available, total);

  setSegment("#segment-reserved", reserved, total);
  setSegment("#segment-weights", weights, total); // # [MOD]
  setSegment("#segment-temporary", temporary, total); // # [MOD]
  setSegment("#segment-allocated", allocated, total);
  setSegment("#segment-available", available, total);

  const numBlocks = resource.num_blocks ?? 0;
  const activeIds = resource.active_block_ids ?? [];
  const cachedIds = resource.cached_block_ids ?? []; // [MOD]
  // [MOD] 数量统计和单block容量拆成两行，避免挤压右侧三态图例。
  document.querySelector("#kv-summary").textContent =
    `${activeIds.length} active · ${cachedIds.length} cached · ${numBlocks} total`;
  document.querySelector("#kv-block-size").textContent =
    `${formatBytes(resource.block_bytes ?? 0)} / block`;
  updateBlockGrid(numBlocks, activeIds, cachedIds);
}

function initPerformanceGrid() {
  const fragment = document.createDocumentFragment();
  for (const [key, label] of PERFORMANCE_METRICS) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.dataset.metric = key;
    card.innerHTML = `<span>${label}</span><strong>--</strong>`;
    fragment.appendChild(card);
  }
  performanceGrid.appendChild(fragment);
}

function updatePerformance(performance) {
  if (!performance) return;
  const count = performance.completed_requests ?? 0;
  document.querySelector("#completed-requests").textContent = `${count} completed`;
  const averages = performance.averages ?? {};

  for (const [key, , kind] of PERFORMANCE_METRICS) {
    const card = performanceGrid.querySelector(`[data-metric="${key}"] strong`);
    const value = averages[key];
    card.textContent = kind === "throughput"
      ? formatThroughput(value)
      : formatLatency(value);
  }

  // [MOD] 全局 TPOT 分位数是所有已完成请求的 request-level TPOT 分布。
  document.querySelector("#performance-cache-hit-rate").textContent = formatRate(performance.cache_hit_rate); // # [MOD]
  const percentiles = performance.tpot_percentiles ?? {};
  for (const key of ["p50", "p95", "p99"]) {
    const node = document.querySelector(`[data-tpot-percentile="${key}"]`);
    node.textContent = formatLatency(percentiles[key]);
  }
}

ws.addEventListener("message", (message) => {
  const event = JSON.parse(message.data);

  if (event.type === "runtime") {
    updateResource(event.resource);
    updatePerformance(event.performance);
    return;
  }

  if (event.type === "accepted") {
    const accepted = new Set(event.accepted_users);
    for (const userId of event.requested_users) {
      if (!accepted.has(userId)) {
        const state = states.get(userId);
        if (state) setBusy(state, false);
      }
    }
    return;
  }

  const state = states.get(event.user_id);
  if (!state) return;

  if (event.type === "token") {
    state.output.textContent = state.outputBase + (event.text ?? "");
    state.output.scrollTop = state.output.scrollHeight;
    return;
  }

  if (event.type === "finished") {
    state.output.textContent = state.outputBase + (event.text ?? "") + "\n\n";
    state.output.scrollTop = state.output.scrollHeight;
    updateUserStatistics(state, event.statistics);
    state.input.value = "";
    setBusy(state, false);
    return;
  }

  if (event.type === "error") {
    state.output.textContent = state.outputBase + `[error] ${event.message}\n\n`;
    state.output.scrollTop = state.output.scrollHeight;
    setBusy(state, false);
  }
});

ws.addEventListener("open", refreshControls);
ws.addEventListener("close", () => {
  for (const state of states.values()) state.button.disabled = true;
  startAll.disabled = true;
});

USERS.forEach(makePanel);
initPerformanceGrid();
refreshControls();
```

## Step 7：CSS 与右侧整列滚动

**修改文件：** `tinyinfer/ui/static/style.css`

将 `.side` 改为整列 `overflow-y: auto` 的 flex column，让 Statistics 和 Controller 一起随右侧滚动，而不是仅 Statistics 内部滚动。保留窄屏断点。

**完整修改后文件：**

```css
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0b0d12;
  color: #edf1f7;
  --panel: rgba(18, 21, 28, .9);
  --panel-soft: #11151d;
  --border: #282e3a;
  --muted: #8c96a8;
  --accent: #dfe7ff;
  --accent-dark: #111827;
  --unused: #565d68;
  --reserved: #c55b3d;
  --allocated: #e59a58;
  --available: #54b985;
  --weights: #8f80cd; /* # [MOD] */
  --temporary: #d3b85b; /* # [MOD] */
  --cached: #6ea8fe; /* [MOD] cached-free KV block */
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(73, 95, 160, .17), transparent 34rem),
    #0b0d12;
}

button, textarea, input { font: inherit; }

.shell {
  width: min(1760px, calc(100vw - 32px));
  height: calc(100vh - 32px);
  margin: 16px auto;
  display: grid;
  grid-template-columns: minmax(0, 2.25fr) minmax(390px, .95fr);
  gap: 14px;
}

.users {
  min-height: 0;
  overflow-y: auto; /* [MOD] 左侧 User 区域独立滚动。 */
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding-right: 4px;
}

.side {
  min-height: 0;
  overflow-y: auto; /* # [MOD] */
  overscroll-behavior: contain; /* # [MOD] */
  display: flex; /* # [MOD] */
  flex-direction: column; /* # [MOD] */
  gap: 14px;
}

.panel {
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
  backdrop-filter: blur(18px);
}

.user-panel {
  flex: 0 0 auto; /* [MOD] 不再被三等分压缩。 */
  min-height: 550px; /* [MOD] panel整体略收紧，output本身改为固定高度。 */
  padding: 16px;
  display: grid;
  grid-template-rows: auto 190px auto; /* [MOD] output行固定高度，生成内容不会撑高panel。 */
  gap: 12px;
}

.panel-head,
.section-title-row,
.kv-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.panel-title {
  color: #f7f9fc;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: .02em;
}

.status,
.subtle,
.request-count,
.kv-key,
.resource-note {
  color: var(--muted);
  font-size: 11px;
}

.status {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 3px 9px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: .02em;
  white-space: nowrap;
}
.status.idle {
  color: #7bd99a;
  background: rgba(84, 185, 133, .10);
  border: 1px solid rgba(84, 185, 133, .24);
}
.status.busy {
  color: #f0aa68;
  background: rgba(229, 154, 88, .10);
  border: 1px solid rgba(229, 154, 88, .25);
}

.output {
  height: 190px; /* [MOD] 固定输出框高度。 */
  min-height: 190px;
  max-height: 190px;
  overflow-y: auto; /* [MOD] 长输出仅在框内滚动。 */
  overscroll-behavior: contain;
  padding: 12px 14px;
  border: 1px solid #232936;
  border-radius: 13px;
  background: #0e1117;
  color: #dce3ee;
  white-space: pre-wrap;
  line-height: 1.5;
  font-size: 13px;
}

.user-lower {
  display: grid;
  grid-template-columns: minmax(0, 1.18fr) minmax(400px, 1.05fr);
  gap: 10px;
  align-items: stretch;
}

.composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 66px;
  gap: 9px;
  align-items: end;
}

textarea {
  width: 100%;
  min-height: 126px; /* [MOD] 左侧可滚动后适当增高输入框。 */
  max-height: 190px;
  resize: vertical;
  border: 1px solid #2b3240;
  border-radius: 13px;
  outline: none;
  padding: 11px 13px;
  background: #151922;
  color: #f5f7fb;
  transition: border-color .15s ease, box-shadow .15s ease, opacity .15s ease;
}

textarea:focus {
  border-color: #657dc8;
  box-shadow: 0 0 0 3px rgba(101, 125, 200, .14);
}

textarea:read-only { color: #aeb6c5; }

button {
  border: 0;
  border-radius: 11px;
  padding: 10px 15px;
  font-weight: 700;
  cursor: pointer;
  transition: transform .12s ease, opacity .12s ease, background .12s ease;
}

button:not(:disabled):active { transform: translateY(1px); }
button:disabled { cursor: default; opacity: .38; }

.primary {
  background: var(--accent);
  color: var(--accent-dark);
}

.user-tools {
  display: grid;
  /* [MOD] Controller优先获得横向空间，Last Request固定为更窄的300px。 */
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 12px;
  min-width: 0;
}

.request-controls,
.user-statistics {
  min-width: 0;
}

.mini-panel {
  border: 1px solid #252b36;
  border-radius: 12px;
  background: #10141b;
  padding: 9px 10px;
}

.mini-title,
.section-label,
.section-title {
  color: #cfd6e2;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.control-row {
  display: grid;
  grid-template-columns: 66px minmax(0, 1fr) 40px;
  gap: 4px;
  align-items: center;
  margin-top: 8px;
  color: #aeb7c6;
  font-size: 10px;
}

.control-value {
  text-align: right;
  color: #eef2f8;
  font-variant-numeric: tabular-nums;
}

input[type="range"] {
  width: 100%;
  min-width: 0; /* [MOD] 允许range在grid单元内正确收缩，不向右溢出。 */
  accent-color: #aabaf0;
}

input:disabled { opacity: .42; }

.toggle-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  color: #aeb7c6;
  font-size: 10px;
}

.switch {
  position: relative;
  width: 34px;
  height: 19px;
  display: inline-block;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider-toggle {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: #3a414e;
  transition: .16s ease;
}

.slider-toggle::after {
  content: "";
  position: absolute;
  width: 13px;
  height: 13px;
  left: 3px;
  top: 3px;
  border-radius: 50%;
  background: #e7ebf2;
  transition: .16s ease;
}

.switch input:checked + .slider-toggle { background: #6c84d1; }
.switch input:checked + .slider-toggle::after { transform: translateX(15px); }
.switch input:disabled + .slider-toggle { opacity: .42; }

.user-stat-grid {
  margin-top: 9px;
  display: grid;
  gap: 8px;
}

.user-stat-row {
  display: grid;
  gap: 12px;
}

.user-stat-row.single {
  grid-template-columns: 1fr;
}

.user-stat-row.pair {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.user-stat {
  min-width: 0;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  font-size: 10px;
  color: #949eae;
  white-space: nowrap;
}

.user-stat span,
.user-stat strong {
  white-space: nowrap;
}

.user-stat strong {
  color: #edf1f7;
  font-variant-numeric: tabular-nums;
}

.statistics-panel {
  padding: 16px;
  flex: 0 0 auto; /* # [MOD] */
  overflow: visible; /* # [MOD] */
}

.global-head { margin-bottom: 12px; }
.live-dot {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #9aa4b4;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.live-dot span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--unused);
  box-shadow: 0 0 12px rgba(84, 185, 133, .55);
}

.stats-section {
  border: 1px solid #252b36;
  border-radius: 14px;
  background: rgba(15, 18, 24, .82);
  padding: 12px;
}
.stats-section + .stats-section { margin-top: 10px; }


.memory-bar {
  height: 17px;
  display: flex;
  overflow: hidden;
  border-radius: 999px;
  margin-top: 9px;
  background: #1c212b;
  border: 1px solid #2c3340;
}
.memory-segment { height: 100%; min-width: 0; transition: width .18s ease; }
.unused { background: var(--unused); }
.weights { background: var(--weights); } /* # [MOD] */
.temporary { background: var(--temporary); } /* # [MOD] */
.reserved { background: var(--reserved); }
.allocated { background: var(--allocated); }
.available { background: var(--available); }

.memory-legend {
  margin-top: 10px;
  display: grid;
  grid-template-columns: 1fr; /* # [MOD] */
  gap: 7px 12px;
}
.memory-legend > div {
  display: grid;
  grid-template-columns: 9px 1fr auto;
  gap: 6px;
  align-items: center;
  color: #9da7b7;
  font-size: 10px;
}
.memory-legend strong { color: #edf1f7; font-variant-numeric: tabular-nums; }

.swatch {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex: 0 0 auto;
}

.kv-head {
  margin-top: 13px;
  align-items: flex-start; /* [MOD] 左侧两行summary与右侧单行legend独立排布。 */
}
.kv-summary-wrap {
  min-width: 0;
  flex: 1 1 auto;
}
.kv-block-size {
  margin-top: 2px; /* [MOD] xx MB / block固定放到下一行。 */
}
.kv-key {
  display: flex;
  align-items: center;
  gap: 9px;
  flex: 0 0 auto;
  flex-wrap: nowrap; /* [MOD] empty/cached/active始终保持一行。 */
  white-space: nowrap;
}
.kv-key-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.block-empty { background: var(--available); } /* [MOD] */
.block-cached { background: var(--cached); } /* [MOD] */
.block-active { background: var(--allocated); } /* [MOD] */

.block-grid {
  margin-top: 8px;
  max-height: 175px;
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(48, minmax(0, 1fr));
  gap: 2px;
  padding: 7px;
  border-radius: 10px;
  border: 1px solid #252b36;
  background: #0c1015;
}

.kv-block {
  aspect-ratio: 1;
  min-width: 0;
  border-radius: 1px;
  opacity: .88;
}
.kv-block.empty { background: var(--available); } /* [MOD] 从未分配 */
.kv-block.cached { background: var(--cached); } /* [MOD] cached-free */
.kv-block.active { background: var(--allocated); opacity: 1; } /* [MOD] active */


.performance-grid {
  margin-top: 10px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
}

.metric-card {
  min-width: 0;
  padding: 8px;
  border-radius: 10px;
  background: #121720;
  border: 1px solid #232a35;
}
.metric-card span {
  display: block;
  color: #8f99aa;
  font-size: 9px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.metric-card strong {
  display: block;
  margin-top: 4px;
  color: #eef2f8;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.percentile-head {
  margin-top: 12px; /* [MOD] */
  color: #9ba6b6;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.percentile-grid {
  margin-top: 7px; /* [MOD] */
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
}

.controller { padding: 16px; }
.controller-grid {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
#start-all { grid-column: span 1; }
.controller-slot {
  display: grid;
  place-items: center;
  min-height: 39px;
  border: 1px dashed #29303d;
  border-radius: 11px;
  color: #4e5664;
  font-size: 10px;
}

@media (max-width: 1250px) {
  .user-lower { grid-template-columns: 1fr; }
  .user-tools { grid-template-columns: minmax(0, 1fr) 300px; } /* [MOD] */
  .output {
    height: 180px; /* [MOD] 窄屏下仍保持固定高度。 */
    min-height: 180px;
    max-height: 180px;
  }
}

@media (max-width: 720px) {
  .user-tools { grid-template-columns: 1fr; }
  .user-statistics { width: 100%; max-width: none; }
}

@media (max-width: 980px) {
  .shell {
    height: auto;
    grid-template-columns: 1fr;
  }
  .users { grid-template-rows: repeat(3, auto); }
  .user-panel { min-height: 430px; }
  .side { overflow: visible; } /* # [MOD] */
}
/* Keep user request controls compact so Last request metrics remain single-line. */
.start-button {
  width: 66px;
  padding-left: 8px;
  padding-right: 8px;
}

.user-statistics {
  width: 300px; /* [MOD] Last Request整体收窄并固定，不再挤压Controller。 */
  max-width: 300px;
}
```

## Step 8：运行验证

先检查 Python 和 JS 语法：

```bash
python -m compileall -q tinyinfer
node --check tinyinfer/ui/static/app.js
```

然后照当前项目已有的 `tinyinfer/ui/app.py` 启动方式运行：

```bash
python -m tinyinfer.ui.app --help
```

请按照实际项目启动参数启动 UI（不要在 `--help` 不能运行时硬套它）。模型加载后观察：

1. 初始化可以出现一轮较明显的 GPU Forward 峰值；之后 dummy KV 应释放，并构造正式 pool。正常推理请求不会再次执行 profile。
2. Resource Statistics 应显示五个分区，总和约为 100%；Num Blocks 与块大小之乘积不超过 KV Budget。显存条中 active/cached-free 状态变化不会更改总 block 数。
3. 第一次输入从未出现的 prompt，Cache Hit Rate 应接近 0%。对有相同完整 token blocks 的 prompt 重试时可能提升，但至少最后一个 prompt token 必须新计算，因此不会保证 100%。
4. 完成两个请求后，全局 Cache Hit Rate 应为**这两条请求各自命中率的平均**，而不是按 token 数加权。
5. 右侧向下滚动时 Statistics、Controller 一同滚动；左侧 User panels 独立滚动；单个输出框继续保持自身滚动。

**实践上的注意事项：** 如果 `max_num_batched_tokens` 很大，在 `torch` Attention backend 的完整 Prefill 上，方阵 attention mask / gather 内存可能偏大，甚至导致启动时 OOM；这表明你所设的最大 batch token 数不适合当前硬件的教学配置，应降低该参数。该方法故意只选一条最大 Prefill，不推断所有 mixed-batch 组合或 CUDA Graph 特性。对于不同 PyTorch、GPU、Attention backend，该 profile 结果也会变化。

### 附加自检

```python
profile = engine.model_runner.resource_profile
snapshot = engine.runtime_resource_snapshot()
assert snapshot["num_blocks"] == profile.num_blocks
assert snapshot["allocated_kv_bytes"] <= profile.num_blocks * profile.block_bytes
assert sum(snapshot[k] for k in (
    "reserved_bytes", "weights_bytes", "temporary_peak_bytes",
    "allocated_kv_bytes", "available_kv_bytes"
)) == snapshot["total_bytes"]
```

以上自检可以在创建 `LLMEngine` 后手动调用；不依赖 UI。
