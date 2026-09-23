# tinyInfer 手把手教程（下）：Day 05–09 —— Qwen3、TP、Loader、CUDA Graph、Benchmark 与原 Day07–09 实践

> 前置：完成上两篇。你已经从空目录搭出了控制面、Sequence/Scheduler、Paged KV allocator、prefix cache、slot mapping、prefill/decode metadata 和教学版 attention。
>
> 本篇的 Day05–06 继续从零搭建模型与执行面；Day07–09 按你的要求**保持原 9-day 讲义内容不改动**，直接作为后半实践环节附在文末。

---

# Day 05：从“runtime shell”接入真正 Qwen3 模型

# 0. 今天最终要补齐的源码树

今天新增：

```text
tinyinfer/
├── layers/
│   ├── activation.py
│   ├── attention.py          # 在 Day04 正确版基础上继续升级
│   ├── embed_head.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── rotary_embedding.py
│   └── sampler.py
├── models/
│   └── qwen3.py
└── utils/
    └── loader.py
```

并把：

```text
engine/model_runner.py
config.py
llm_engine.py
```

从教学接口升级为真实模型路径。

今天结束后，你应该能完整解释：

```text
HF safetensors
  ↓ custom loader
Qwen3 custom modules
  ↓
ModelRunner
  ↓ prepare prefill/decode
Attention + Paged KV
  ↓
LM Head
  ↓
Sampler
  ↓
next token
```

---

# 1. Step 1：先搭最基础 Linear，再谈 TP

创建：

```text
tinyinfer/layers/linear.py
```

第一版：

```python
import torch
import torch.nn.functional as F
from torch import nn


class LinearBase(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
            if bias else None
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

这里你不要直接写 `nn.Linear`，是因为后面 loader/TP 需要给 parameter 挂自己的 loading contract。

增加：

```python
def default_weight_loader(param, loaded_weight, shard_id=None):
    if shard_id is not None:
        raise ValueError("base parameter does not accept shard_id")
    param.data.copy_(loaded_weight)
```

构造时：

```python
self.weight.weight_loader = default_weight_loader
```

这看起来只是多一层函数，却是 custom loader 能统一处理普通参数和 packed/TP 参数的关键。

---

# 2. Step 2：Column Parallel 与 Row Parallel 自己推一遍

线性层：

```text
Y = XW^T
```

## Column Parallel

按 output features 切：

```text
W = [W0; W1; ...]
```

每卡：

```text
Yi = X Wi^T
```

得到不同输出 feature shard。

## Row Parallel

按 input features 切：

```text
W = [W0, W1, ...]
X = [X0, X1, ...]
```

每卡：

```text
partial_i = Xi Wi^T
```

最后：

```text
Y = sum_i partial_i
```

所以 row-parallel 常需要 all-reduce。

---

# 3. Step 3：先写 TP helper，再写并行 Linear

在 `linear.py`：

```python
import torch.distributed as dist


def tp_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def tp_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()
```

Column Parallel：

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, in_features, out_features, bias=False, **kwargs):
        world = tp_world_size()
        if out_features % world != 0:
            raise ValueError("out_features must divide TP world size")
        self.global_out_features = out_features
        super().__init__(in_features, out_features // world, bias=bias, **kwargs)

        def loader(param, loaded_weight, shard_id=None):
            rank = tp_rank()
            shard = loaded_weight.chunk(tp_world_size(), dim=0)[rank]
            param.data.copy_(shard)

        self.weight.weight_loader = loader
```

Row Parallel：

```python
class RowParallelLinear(LinearBase):
    def __init__(self, in_features, out_features, bias=False, reduce_results=True, **kwargs):
        world = tp_world_size()
        if in_features % world != 0:
            raise ValueError("in_features must divide TP world size")
        self.global_in_features = in_features
        self.reduce_results = reduce_results
        super().__init__(in_features // world, out_features, bias=bias, **kwargs)

        def loader(param, loaded_weight, shard_id=None):
            rank = tp_rank()
            shard = loaded_weight.chunk(tp_world_size(), dim=1)[rank]
            param.data.copy_(shard)

        self.weight.weight_loader = loader

    def forward(self, x):
        y = super().forward(x)
        if self.reduce_results and tp_world_size() > 1:
            dist.all_reduce(y)
        return y
```

单卡 RTX 5090 上无法实测 TP 性能，所以今天只做：

```text
数学理解 + 单卡 world_size=1 correctness
```

不要在简历里声称多卡 TP 优化。

---

# 4. Step 4：实现 packed QKV / Gate-Up projection

Qwen 系列常把多个 projection packed：

```text
q_proj
k_proj
v_proj
```

→ 一个：

```text
qkv_proj
```

以及：

```text
gate_proj
up_proj
```

→ 一个：

```text
gate_up_proj
```

这样可以减少 kernel launch / improve GEMM granularity。

创建一个可按 shard id 加载的 packed linear：

```python
class PackedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, in_features, shard_out_features, shard_names, **kwargs):
        self.shard_out_features = list(shard_out_features)
        self.shard_names = list(shard_names)
        super().__init__(in_features, sum(shard_out_features), **kwargs)

        def loader(param, loaded_weight, shard_id=None):
            if shard_id is None:
                raise ValueError("packed parameter requires shard_id")

            idx = self.shard_names.index(shard_id)
            offsets = [0]
            for n in self.shard_out_features:
                offsets.append(offsets[-1] + n)

            world = tp_world_size()
            rank = tp_rank()
            local_begin = offsets[idx] // world
            local_size = self.shard_out_features[idx] // world

            shard = loaded_weight.chunk(world, dim=0)[rank]
            param.data[local_begin:local_begin + local_size].copy_(shard)

        self.weight.weight_loader = loader
```

教学重点：packed projection 改变的是**parameter layout / kernel granularity**，不是 transformer 数学。

---

# 5. Step 5：实现 RMSNorm

创建：

```text
tinyinfer/layers/layernorm.py
```

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            new_residual = x
        else:
            new_residual = x

        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = x.float() * torch.rsqrt(variance + self.eps)
        y = y.to(dtype=x.dtype) * self.weight
        return y, new_residual
```

为什么返回 residual？

因为高性能实现常把：

```text
residual add + norm
```

融合，减少 memory traffic。我们的 API 先为后续 fusion 留出结构。

---

# 6. Step 6：实现 SiLU-and-Mul / SwiGLU

创建：

```text
tinyinfer/layers/activation.py
```

```python
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

对应 MLP：

```text
x
 ↓ packed gate_up_proj
[gate | up]
 ↓ SiLU(gate) * up
 ↓ down_proj
```

---

# 7. Step 7：手写 RoPE

创建：

```text
tinyinfer/layers/rotary_embedding.py
```

```python
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position, theta=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        positions = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, positions, q, k):
        cos = self.cos[positions].repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = self.sin[positions].repeat_interleave(2, dim=-1).unsqueeze(1)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k
```

为什么 ModelRunner 必须生成 `positions`？

因为 RoPE 与 absolute token position 绑定；prefix cache hit 后 suffix 从 position `num_cached_tokens` 开始，不能重新从 0 编号。

---

# 8. Step 8：实现 Sampler

创建：

```text
tinyinfer/layers/sampler.py
```

先保持和原 Day08 baseline 一致的思路：

```python
import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float() / temperatures.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
```

这里先用最易懂版本；Day08 保持原实践，再把它改成 logit-space equivalent sampling。

---

# 9. Step 9：Vocab Parallel Embedding / LM Head

创建：

```text
tinyinfer/layers/embed_head.py
```

教学版先保证 world=1 正确，再支持 world>1：

```python
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from tinyinfer.layers.linear import tp_rank, tp_world_size


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, dtype=None, device=None):
        super().__init__()
        world = tp_world_size()
        if vocab_size % world != 0:
            raise ValueError("vocab_size must divide TP world size in teaching implementation")

        self.vocab_size = vocab_size
        self.local_vocab = vocab_size // world
        self.vocab_start = tp_rank() * self.local_vocab
        self.vocab_end = self.vocab_start + self.local_vocab

        self.weight = nn.Parameter(
            torch.empty(self.local_vocab, hidden_size, dtype=dtype, device=device)
        )

        def loader(param, loaded_weight, shard_id=None):
            rank = tp_rank()
            param.data.copy_(loaded_weight.chunk(tp_world_size(), dim=0)[rank])

        self.weight.weight_loader = loader

    def forward(self, input_ids):
        if tp_world_size() == 1:
            return F.embedding(input_ids, self.weight)

        mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = input_ids - self.vocab_start
        local_ids = local_ids.masked_fill(mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        dist.all_reduce(out)
        return out


class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, hidden):
        local_logits = F.linear(hidden, self.weight)
        if tp_world_size() == 1:
            return local_logits

        gathered = [torch.empty_like(local_logits) for _ in range(tp_world_size())]
        dist.all_gather(gathered, local_logits)
        return torch.cat(gathered, dim=-1)
```

---

# 10. Step 10：组装 Qwen3 Attention

创建：

```text
tinyinfer/models/qwen3.py
```

先建立 config aliases：

```python
import math
import torch
from torch import nn

from tinyinfer.layers.activation import SiluAndMul
from tinyinfer.layers.attention import Attention
from tinyinfer.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from tinyinfer.layers.layernorm import RMSNorm
from tinyinfer.layers.linear import PackedColumnParallelLinear, RowParallelLinear
from tinyinfer.layers.rotary_embedding import RotaryEmbedding
from tinyinfer.layers.sampler import Sampler
```

Attention module：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx, kv_cache):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = PackedColumnParallelLinear(
            self.hidden_size,
            [q_size, kv_size, kv_size],
            ["q", "k", "v"],
            bias=False,
        )
        self.o_proj = RowParallelLinear(q_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.attn = Attention(
            layer_idx=layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim),
            kv_cache=kv_cache,
            block_size=kv_cache.storage.shape[3],
        )

    def forward(self, x, positions):
        qkv = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary(positions, q, k)
        out = self.attn(q, k, v)
        out = out.reshape(-1, q_size)
        return self.o_proj(out)
```

> 如果 Qwen3 checkpoint 的具体 q/k norm、bias、head_dim 字段与你当前 transformers revision 不同，以本地 `config.json` 为准。教程重点是模块关系；最终权重名必须通过 loader mapping 与 checkpoint 对齐。

---

# 11. Step 11：组装 MLP 与 Decoder Layer

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = PackedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            ["gate", "up"],
            bias=False,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x):
        return self.down_proj(self.act(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, kv_cache):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_idx, kv_cache)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, positions):
        normed, residual = self.input_layernorm(x)
        x = self.self_attn(normed, positions)
        normed, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(normed)
        return x, residual
```

最后一层 residual 合并：

```text
final hidden = residual + branch
→ final RMSNorm
```

---

# 12. Step 12：组装整个 Qwen3ForCausalLM

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        ".q_proj.": (".qkv_proj.", "q"),
        ".k_proj.": (".qkv_proj.", "k"),
        ".v_proj.": (".qkv_proj.", "v"),
        ".gate_proj.": (".gate_up_proj.", "gate"),
        ".up_proj.": (".gate_up_proj.", "up"),
    }

    def __init__(self, config, kv_cache):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i, kv_cache)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler()

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            if residual is not None:
                # decoder layer's fused-residual API can be refined; this teaching
                # implementation keeps the state explicit.
                pass
            x, residual = layer(x, positions)

        x, _ = self.norm(x, residual)
        return x

    def compute_logits(self, hidden):
        return self.lm_head(hidden)

    def sample(self, logits, temperatures):
        return self.sampler(logits, temperatures)
```

这里建议你**不要直接相信第一版 residual API**。马上写一个 small-tensor unit test，与 HuggingFace 单层/整模型 hidden state 对齐。手搭模型最重要的是逐层 correctness，而不是先追求“能生成文本”。

---

# 13. Step 13：写 Safetensors Loader

创建：

```text
tinyinfer/utils/loader.py
```

```python
import os
from glob import glob

from safetensors import safe_open


def load_model(model, path: str):
    mapping = getattr(model, "packed_modules_mapping", {})
    loaded = set()

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors found under {path}")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                handled = False

                for old_name, (packed_name, shard_id) in mapping.items():
                    if old_name in weight_name:
                        param_name = weight_name.replace(old_name, packed_name)
                        param = model.get_parameter(param_name)
                        loader = getattr(param, "weight_loader", None)
                        if loader is None:
                            raise RuntimeError(f"{param_name} has no custom weight_loader")
                        loader(param, f.get_tensor(weight_name), shard_id)
                        loaded.add(param_name)
                        handled = True
                        break

                if handled:
                    continue

                try:
                    param = model.get_parameter(weight_name)
                except AttributeError:
                    # tied/extra checkpoint tensors需按具体模型判断；
                    # 学习阶段不要静默吞掉所有 mismatch。
                    continue

                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    param.data.copy_(f.get_tensor(weight_name))
                else:
                    loader(param, f.get_tensor(weight_name))
                loaded.add(weight_name)

    return loaded
```

Day06 你要加入“未加载参数/未消费 checkpoint tensor”检查，避免 silent partial load。

---

# 14. Step 14：Config 终于连接 HF model config

修改 `config.py`：

```python
import os
from transformers import AutoConfig, AutoTokenizer
```

增加：

```python
def load_model_metadata(self):
    if self.model is None:
        raise ValueError("model path is required for real ModelRunner")
    if not os.path.isdir(self.model):
        raise FileNotFoundError(self.model)

    self.hf_config = AutoConfig.from_pretrained(self.model)
    self.max_model_len = min(
        self.max_model_len,
        self.hf_config.max_position_embeddings,
    )

    tokenizer = AutoTokenizer.from_pretrained(self.model)
    if tokenizer.eos_token_id is not None:
        self.eos_token_id = tokenizer.eos_token_id
```

这时才让 runtime config 与 model config 汇合。

---

# 15. Step 15：ModelRunner 初始化顺序——为什么 warmup 在 KV cache allocation 之前？

真实路径应该是：

```text
1. init CUDA/device/TP
2. build model
3. load weights
4. warmup model
5. measure peak non-KV memory
6. compute num KV blocks
7. allocate KV cache
8. optional capture CUDA Graph
```

但是 attention model forward 又需要 KV cache。教学实现可以先给 warmup 用一个临时 cache 或把 profile run 的 attention backend设成 no-cache；真正接近 nano-vLLM 时，再按照其具体初始化技巧实现。

关键概念：

```text
如果先把“当前剩余显存”全拿去 KV cache
↓
第一次真实 forward 还会产生 activation/workspace
↓
可能 OOM
```

所以 profile/warmup 的价值是估计：

```text
weights + activation/workspace peak
```

然后：

```text
KV bytes budget
= total_gpu_memory * utilization
- measured_non_kv_peak
```

---

# 16. Step 16：自己实现 KV block capacity 计算

假设 HF config：

```python
num_layers = config.hf_config.num_hidden_layers
num_kv_heads = config.hf_config.num_key_value_heads
head_dim = getattr(
    config.hf_config,
    "head_dim",
    config.hf_config.hidden_size // config.hf_config.num_attention_heads,
)
```

每 physical block bytes：

```python
bytes_per_elem = 2  # BF16
block_bytes = (
    num_layers
    * 2
    * config.kvcache_block_size
    * num_kv_heads
    * head_dim
    * bytes_per_elem
)
```

可分 block：

```python
num_blocks = int(kv_budget_bytes // block_bytes)
```

然后把结果写回：

```python
config.num_kvcache_blocks = num_blocks
```

这时 Day03 的 BlockManager 不再使用拍脑袋 128，而是由实际 GPU capacity 决定。

---

# 17. Step 17：主动复现你之前看到的 warmup Sequence ID 现象

为了完全吃透之前的疑问，写一个 warmup path，**故意使用 Sequence** 生成最大 workload：

```python
def warmup_sequences(self):
    seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
    num_seqs = min(
        max(1, self.config.max_num_batched_tokens // seq_len),
        self.config.max_num_seqs,
    )
    return [
        Sequence([0] * seq_len, SamplingParams(temperature=1.0, max_tokens=1))
        for _ in range(num_seqs)
    ]
```

然后：

```python
seqs = self.warmup_sequences()
print("warmup ids:", [s.seq_id for s in seqs])
```

如果 warmup 创建 4 个：

```text
warmup ids: [0,1,2,3]
```

随后真实请求自然：

```text
[4,5,6,...]
```

这一次你是亲手造成并解释这个现象，而不是被 upstream 日志吓到。

---

# 18. Step 18：Tokenizer 接入 LLMEngine

到现在才把字符串 API 加回去。

`llm_engine.py`：

```python
from transformers import AutoTokenizer
```

初始化：

```python
self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
```

新增：

```python
def generate(self, prompts, sampling_params, use_tqdm=False):
    if isinstance(prompts, str):
        prompts = [prompts]

    encoded = [
        self.tokenizer.encode(p, add_special_tokens=False)
        for p in prompts
    ]

    outputs = self.generate_token_ids(encoded, sampling_params)

    for out in outputs:
        out["text"] = self.tokenizer.decode(out["token_ids"])
    return outputs
```

这时候公共调用链才最终成为：

```text
string prompt
→ tokenizer
→ Sequence
→ Scheduler
→ BlockManager
→ ModelRunner
→ Qwen3
→ Attention/KV
→ Sampler
→ token ids
→ tokenizer.decode
```

---

# 19. Step 19：CUDA Graph —— 先理解问题，再写 capture

Decode iteration 常见特征：

```text
小 batch
短 kernel
每轮 shape 相对稳定
iteration 很多
```

因此 CPU launch overhead 占比可能不小。

CUDA Graph 的目的不是：

```text
让 GEMM 本身算得更快
```

而是：

```text
把一串固定 CUDA operations capture
→ 后续一次 replay
→ 降低 Python/CPU launch overhead
```

---

# 20. Step 20：为什么要固定 buffer？

Graph replay 要求地址/shape 尽量稳定，所以准备：

```python
self.graph_input_ids = torch.empty(max_graph_batch, dtype=torch.long, device="cuda")
self.graph_positions = torch.empty(max_graph_batch, dtype=torch.long, device="cuda")
self.graph_slots = torch.empty(max_graph_batch, dtype=torch.long, device="cuda")
```

每轮：

```text
copy new data into static buffers
↓
replay graph
```

而不是每轮创建新 tensor address。

---

# 21. Step 21：写最小 CUDA Graph wrapper

概念代码：

```python
class DecodeGraph:
    def __init__(self, model, batch_size, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.input_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None

    def capture(self):
        # 先 warmup 同 shape 若干次
        for _ in range(3):
            _ = self.model(self.input_ids, self.positions)
        torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            self.output = self.model(self.input_ids, self.positions)

    def replay(self, input_ids, positions):
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.graph.replay()
        return self.output
```

真实 runtime 还必须把：

```text
slot_mapping
context_lens
block_tables
```

也放进 static/reusable buffers，并处理多种 batch size graph。

这也是为什么 CUDA Graph 是 runtime feature，而不是模型层 feature。

---

# 22. Step 22：为什么 Prefill 不优先走同一套 Graph？

Prefill shape 变化很大：

```text
request A: 20 tokens
request B: 2000 tokens
下一轮: 300+400+17 tokens
```

Capture 太多 shape 会导致：

```text
graph explosion
memory overhead
management complexity
```

Decode 则常可按 batch size bucket：

```text
1,2,4,8,16,...
```

更适合 graph replay。

---

# 23. Step 23：Day05 最重要的 correctness 测试——和 HuggingFace 对齐

不要先看生成文本“像不像人话”。

先做：

```text
相同 checkpoint
相同 input_ids
相同 positions
关闭 sampling randomness
比较 logits
```

建议写：

```text
tests/test_qwen3_logits.py
```

思路：

```python
with torch.no_grad():
    hf_logits = hf_model(input_ids).logits[:, -1]
    tiny_logits = ...

max_abs = (hf_logits - tiny_logits).abs().max().item()
print(max_abs)
```

逐级定位：

```text
embedding
→ layer0 attn
→ layer0 mlp
→ layerN
→ final norm
→ lm head
```

如果一上来只比最终文本，任何一个层的误差都难定位。

建议 commit：

```bash
git add .
git commit -m "day05: add qwen3 execution stack loader TP and decode graph"
```

---

# Day 06：把“复现框架”升级成“能评测、能定位、能修改的工程”

# 24. 今天不再添加大组件，而是完成工程闭环

Day06 有四个目标：

```text
A. 正确性验证体系
B. benchmark 体系
C. trace/profiling 体系
D. 至少 2 个属于你自己的改进点
```

这比再加一个花哨功能更重要。

---

# 25. Step 1：整理最终 tinyInfer 目录树

到这里应接近：

```text
tinyInfer/
├── README.md
├── pyproject.toml
├── examples/
│   ├── basic_generate.py
│   ├── day02_scheduler_trace.py
│   └── day04_metadata_dump.py
├── benchmarks/
│   ├── throughput.py
│   ├── latency.py
│   └── eager_vs_graph.py
├── tests/
│   ├── test_sequence.py
│   ├── test_scheduler.py
│   ├── test_block_manager.py
│   ├── test_attention_meta.py
│   ├── test_sampler.py
│   └── test_qwen3_logits.py
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
    │   ├── activation.py
    │   ├── attention.py
    │   ├── embed_head.py
    │   ├── layernorm.py
    │   ├── linear.py
    │   ├── rotary_embedding.py
    │   └── sampler.py
    ├── models/
    │   └── qwen3.py
    └── utils/
        ├── context.py
        └── loader.py
```

这已经覆盖 nano-vLLM 的核心组件边界。

---

# 26. Step 2：写 benchmark contract，而不是随手 `time.time()`

创建：

```text
benchmarks/throughput.py
```

至少记录：

```text
num_requests
prompt_tokens
output_tokens
wall_time
output_tokens/s
total_tokens/s
```

GPU benchmark 需要：

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
...
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

否则 CPU 提前返回会低估时间。

---

# 27. Step 3：加入 TTFT / E2E 基础 timestamp

在 Sequence 增加：

```python
import time

self.arrival_time = time.perf_counter()
self.first_token_time = None
self.finish_time = None
```

在第一次 append generated token：

```python
if self.num_completion_tokens == 0:
    self.first_token_time = time.perf_counter()
```

finish：

```python
self.finish_time = time.perf_counter()
```

于是：

```python
TTFT = first_token_time - arrival_time
E2E  = finish_time - arrival_time
```

Day07 会在这个基础上正式做 scheduler latency benchmark；今天只把 observability 铺好。

---

# 28. Step 4：小创新 1 —— Runtime invariant checker

把 Day03 的 allocator checker 扩展为整个 runtime checker：

```python
def check_runtime_invariants(self):
    ids = [s.seq_id for s in self.scheduler.waiting] + [s.seq_id for s in self.scheduler.running]
    assert len(ids) == len(set(ids))

    for seq in self.scheduler.running:
        assert seq.block_table
        assert seq.num_cached_tokens <= seq.num_tokens
        assert len(seq.block_table) >= seq.num_blocks - 1

    self.scheduler.block_manager.check_consistency()
```

只在：

```bash
TINYINFER_DEBUG=1
```

时开启。

这是很实用的工程增强：复杂 scheduler/allocator bug 往往不是当场崩，而是几轮后才表现为错误 token 或 OOM。

---

# 29. Step 5：小创新 2 —— 分层 trace event，而不是散落 print

创建：

```text
tinyinfer/trace.py
```

```python
import json
import os
import time


def emit(event: str, **kwargs):
    if os.getenv("TINYINFER_TRACE", "0") != "1":
        return
    record = {
        "ts": time.perf_counter(),
        "event": event,
        **kwargs,
    }
    print("[tinyinfer-trace] " + json.dumps(record, ensure_ascii=False))
```

在关键位置：

```text
request_add
schedule_begin
schedule_end
kv_alloc
kv_free
prepare_prefill
prepare_decode
model_begin
model_end
sample
request_finish
```

都 emit JSON。

这样后面可以写脚本自动统计：

```text
queueing time
prefill time
decode iterations
block usage
```

这比普通 `[sched] print` 更接近真实 systems observability。

---

# 30. Step 6：小创新 3（可选）—— Strict Loader

你自己的 loader 不应该静默接受：

```text
checkpoint 有 tensor 没加载
model 有 parameter 没初始化
```

增加：

```text
loaded_checkpoint_names
loaded_parameter_names
expected_parameter_names
```

最后比较：

```python
missing = expected - loaded
if missing:
    raise RuntimeError(f"unloaded parameters: {sorted(missing)[:20]}")
```

这类改进比“随便重写一个 kernel”更容易在面试里讲清楚工程价值。

---

# 31. Step 7：benchmark matrix

至少做下面几组：

```text
A. prompt length: 32 / 256 / 1024 / 2048
B. output length: 16 / 64 / 256
C. concurrent requests: 1 / 4 / 16
D. eager vs CUDA graph
E. prefix share: 0% / one full block / multi-block
```

输出 CSV：

```text
benchmark, prompt_len, output_len, batch, ttft_ms, e2e_ms, output_tok_s, peak_mem_mb
```

不要只保存截图。

---

# 32. Step 8：和 upstream nano-vLLM 做“结构对照”，不是追求逐行相同

现在再对照：

```text
nano-vllm                         tinyInfer
------------------------------------------------------
config.py                         config.py
sampling_params.py                sampling_params.py
engine/sequence.py                engine/sequence.py
engine/scheduler.py               engine/scheduler.py
engine/block_manager.py           engine/block_manager.py
engine/model_runner.py            engine/model_runner.py
engine/llm_engine.py              engine/llm_engine.py
layers/attention.py               layers/attention.py
layers/linear.py                  layers/linear.py
layers/embed_head.py              layers/embed_head.py
layers/rotary_embedding.py        layers/rotary_embedding.py
layers/layernorm.py               layers/layernorm.py
layers/activation.py              layers/activation.py
layers/sampler.py                 layers/sampler.py
models/qwen3.py                   models/qwen3.py
utils/context.py                  utils/context.py
utils/loader.py                   utils/loader.py
```

你应该能够对每个文件回答：

```text
如果删掉它，哪个责任会无家可归？
```

这才叫真正复现框架。

---

# 33. Step 9：对照真实 nano-vLLM 时重点看“它比你多做了什么”

你自己的版本有意先从正确性出发，因此上游通常会多出：

```text
更高效的 FlashAttention backend
Triton KV cache write
更紧凑的 tensor packing
TP multiprocessing/IPC
CUDA Graph batch buckets
更成熟的 weight shard loader
更细的 prefix cache allocator semantics
```

学习方法：

```text
先在 tinyInfer 中知道“为什么要有这个接口”
再看 upstream “如何把它做快”
```

而不是倒过来死记一串优化代码。

---

# 34. Step 10：Day01–06 最终白板题

不看代码画：

```text
Prompt string
  ↓ tokenizer
Sequence
  ↓ waiting queue
Scheduler
  ├─ token budget
  ├─ seq budget
  └─ KV block availability
  ↓
BlockManager
  ├─ physical block pool
  ├─ prefix hash
  └─ ref count
  ↓
ModelRunner
  ├─ prepare_prefill
  │   ├─ packed input ids
  │   ├─ positions
  │   ├─ cu_seqlens
  │   └─ slot_mapping
  └─ prepare_decode
      ├─ one token / seq
      ├─ context_lens
      ├─ block_tables
      └─ slot_mapping
  ↓
Qwen3
  ├─ embedding
  ├─ decoder layers
  │   ├─ RMSNorm
  │   ├─ QKV projection
  │   ├─ RoPE
  │   ├─ Attention + paged KV
  │   └─ SwiGLU MLP
  ├─ final norm
  └─ LM head
  ↓
Sampler
  ↓ next token
Scheduler.postprocess
  ├─ append
  ├─ finish?
  └─ free KV
```

如果你可以自己解释这张图，Day07–09 的“自己修改 Scheduler / Sampler / SpecDecode”才是真正建立在系统理解上的实践，而不是照着 patch 改几行。

---

# 35. Day06 最终 commit 建议

```bash
git add .
git commit -m "day06: complete tinyInfer runtime benchmarks tracing and invariants"
```

到这里，前 6 天从“读 nano-vLLM”彻底改造成了：

```text
Day01 亲手造 Engine loop
Day02 亲手造 Sequence + Scheduler + continuous batching
Day03 亲手造 Paged KV + prefix cache allocator
Day04 亲手造 ModelRunner metadata + attention/KV path
Day05 亲手造 Qwen3 layers + loader + TP + CUDA graph
Day06 亲手做 correctness / benchmark / trace / 自己的工程增强
```

下面按你的要求，保留原 Day07–09 实践环节，不改内容。

---



---

# 原 Day07（保持不变）

# `day07-scheduler-latency-and-fair-prefill.md`：从“看懂 Scheduler”到“做一次可量化的调度优化”

> 前置：完整完成 Day 01~06；能解释 `Sequence -> Scheduler -> BlockManager -> ModelRunner`。  
> 今日目标：在**不重写 nano-vLLM 架构**的前提下，对长短 prompt 混合负载做一次真实、可复现、可回退的调度实验：补齐 TTFT/E2E 指标，增加可配置的 prefill chunk 上限，并尝试 round-robin 公平化。  
> 今日成果边界：这是“调度策略改造 + benchmark”，不是声称发明新的 Chunked Prefill。当前 upstream 已经能把首个 waiting request 的 prefill 按 token budget 切块；我们做的是**参数化 + 公平性实验 + 延迟量化**。

---

# 0. 今天为什么值得写进简历？

推理系统面试经常问：

```text
一个 8K prompt 和很多 128-token prompt 同时到达，
如何避免长 prefill 把短请求的 TTFT 拖垮？
```

只回答“Continuous Batching / Chunked Prefill”不够。今天要真正做到：

```text
读 scheduler.py
    ↓
构造长短请求混合 workload
    ↓
测 baseline TTFT / E2E / throughput
    ↓
改调度策略
    ↓
再次测量
    ↓
解释 throughput 与 tail latency trade-off
```

这是一条完整的工程闭环。

---

# 1. 先确认当前 upstream 到底做了什么

在仓库根目录：

```bash
sed -n '1,130p' nanovllm/engine/scheduler.py
```

你应该重点读这几个字段：

```python
self.max_num_seqs
self.max_num_batched_tokens
self.waiting
self.running
```

当前 prefill 核心逻辑可概括为：

```python
remaining = max_num_batched_tokens - num_batched_tokens
num_tokens = seq.num_tokens - seq.num_cached_tokens
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

并且当前代码有一个关键限制：

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

也就是：

```text
只有本轮第一个 prefill request 可以被 chunk
后续 request 如果塞不下，则直接结束本轮 prefill scheduling
```

同时，如果一个请求还没有完成 prefill，`postprocess()` 会增加 `num_cached_tokens`，但不会 `append_token()`，于是它仍停留在 waiting 队列中。

先回答下面三个问题再继续：

```text
Q1. prefill progress 存在哪里？
A1. seq.num_cached_tokens

Q2. 本轮要算多少新 token 存在哪里？
A2. seq.num_scheduled_tokens

Q3. 尚未完成 prefill 的 seq 在哪里？
A3. 仍在 scheduler.waiting
```

---

# 2. 今天的目录树变化

先建分支：

```bash
git status
git switch -c feat/fair-prefill-bench
```

新增实验目录：

```bash
mkdir -p learning/bench learning/results/day07
```

最终目录变化：

```text
nano-vllm/
├── nanovllm/
│   ├── config.py                       # 修改：prefill_chunk_size
│   └── engine/
│       ├── sequence.py                 # 修改：轻量 timing metadata
│       └── scheduler.py                # 修改：chunk cap + 可选 RR
└── learning/
    ├── bench/
    │   └── 07_scheduler_latency.py     # 新增
    └── results/day07/                  # 新增
```

原则：

```text
先只加观测，不改算法；
baseline 跑通后再改调度；
每个阶段都能 git diff / git checkout 回退。
```

---

# 3. Step 1：先补观测指标，不要先优化

## 3.1 为什么不能只看 tokens/s？

今天至少区分：

```text
TTFT = first_token_time - arrival_time
E2E  = finish_time - arrival_time
Throughput = total completion tokens / wall time
```

长 prefill 更容易伤害其他请求的 **TTFT**，而不仅仅是整体吞吐。

## 3.2 修改 `nanovllm/engine/sequence.py`

先查看原文件：

```bash
sed -n '1,220p' nanovllm/engine/sequence.py
```

顶部增加：

```python
from time import perf_counter
```

在 `Sequence.__init__()` 的末尾增加：

```python
self.arrival_time = perf_counter()
self.first_token_time = None
self.finish_time = None
```

找到 `append_token()`，在真正 append completion token 前加入：

```python
if self.first_token_time is None:
    self.first_token_time = perf_counter()
```

不要在 prefill 时设置 `first_token_time`。理由：

```text
prefill 只是在处理 prompt，
真正的 TTFT 要等第一个 completion token 产生。
```

## 3.3 在 finish 时写入时间戳

打开：

```text
nanovllm/engine/scheduler.py
```

在判断请求完成的分支中，设置：

```python
from time import perf_counter
```

然后：

```python
if (not seq.ignore_eos and token_id == self.eos) or \
        seq.num_completion_tokens == seq.max_tokens:
    seq.finish_time = perf_counter()
    seq.status = SequenceStatus.FINISHED
    ...
```

先运行原有 example / benchmark，确保没有行为变化。

---

# 4. Step 2：写 baseline workload

创建：

```text
learning/bench/07_scheduler_latency.py
```

这一版不要一开始追求完整 benchmark framework，只需要能构造：

```text
1 个长 prompt
+ N 个短 prompt
```

推荐第一组：

```text
long_prompt  ≈ 4096 tokens
short_prompt ≈ 128 tokens
short_count  = 8
max_tokens   = 32
```

如果直接用自然语言很难精确 token 数，可以用 tokenizer 重复一段普通文本直到达到目标 token 数，再 decode 回字符串。

伪代码结构：

```python
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

MODEL = ".../Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def make_prompt(n_tokens: int):
    seed = "Explain GPU inference optimization briefly. "
    ids = tokenizer.encode(seed * (n_tokens // 5 + 10), add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens])


prompts = [make_prompt(4096)] + [make_prompt(128) for _ in range(8)]
params = SamplingParams(temperature=0.0, max_tokens=32)

llm = LLM(MODEL, enforce_eager=True, max_model_len=8192)
outputs = llm.generate(prompts, params)
```

> 注意：如果你本地 `SamplingParams` / `LLM` 参数名与这里不同，以你 Day 01~06 已跑通的接口为准，不要机械复制。

今天 baseline 最少保存：

```text
variant,prompt_tokens,request_id,ttft_ms,e2e_ms,completion_tokens
baseline,4096,0,...,...,...
baseline,128,1,...,...,...
...
```

如果 public API 暂时拿不到 Sequence timing，可以临时在 engine 输出结构中附带调试字段，或者在 `Scheduler.postprocess()` 写日志。优先保持改动小。

---

# 5. Step 3：给 prefill 加“显式 chunk cap”

当前 upstream 的 chunk 大小实际上由：

```text
max_num_batched_tokens - 本轮已使用 tokens
```

隐式决定。

为了能做控制变量实验，我们加：

```text
prefill_chunk_size
```

## 5.1 修改 `nanovllm/config.py`

先查看 Config 定义：

```bash
sed -n '1,240p' nanovllm/config.py
```

在 `Config` 里增加类似：

```python
prefill_chunk_size: int = 512
fair_prefill: bool = False
```

并做最小校验：

```python
assert self.prefill_chunk_size > 0
```

如果当前 Config 使用 `__post_init__`，把校验放进去。

## 5.2 修改 Scheduler 初始化

```python
self.prefill_chunk_size = config.prefill_chunk_size
self.fair_prefill = config.fair_prefill
```

## 5.3 修改本轮 token 数

原来：

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

改成：

```python
seq.num_scheduled_tokens = min(
    num_tokens,
    remaining,
    self.prefill_chunk_size,
)
```

先只做这一处，不做 round-robin。

### 你刚刚实现的是什么？

```text
固定上限 Chunked Prefill
```

它保证即使：

```text
max_num_batched_tokens = 4096
```

一个长 request 也不会一次吃满整个 4096-token budget，而最多吃：

```text
prefill_chunk_size
```

但此时它仍可能一直待在 `waiting[0]`。

---

# 6. Step 4：为什么仅仅 chunk 还不一定公平？

假设：

```text
waiting = [Long, Short1, Short2, Short3]
```

Long 每轮只处理 512 tokens，但没完成时仍在 `waiting[0]`。

下一轮：

```text
waiting = [Long, Short1, Short2, Short3]
```

于是 Long 仍然最先拿 quota。

所以需要第二个实验变量：

```text
完成一个 partial prefill chunk 后，
是否把该 seq rotate 到 waiting 尾部？
```

---

# 7. Step 5：实现最小 round-robin partial prefill

这里不要重写整个 Scheduler；只对“本轮 chunk 完成但整个 prompt 未完成”的 request 做轮转。

当前 `schedule()` 返回后，`postprocess()` 已经知道：

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
```

然后：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

可以将这个分支扩展为：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    if self.fair_prefill:
        assert self.waiting and self.waiting[0] is seq
        self.waiting.popleft()
        self.waiting.append(seq)
    continue
```

但先别盲目复制。你必须确认两个不变量：

```text
1. 这个 seq 当前确实仍在 waiting；
2. 它确实仍是 waiting[0]。
```

如果当前实现一次 schedule 允许多个 prefill request，第二个不变量不一定对所有 seq 都成立。

更稳妥的教学实现是：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    if self.fair_prefill:
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass
        self.waiting.append(seq)
    continue
```

因为 deque 的 `remove` 是 O(n)，这不是 production 最优实现，但 Day 07 的目的是验证 scheduling policy；请求数几十级时足够。

面试时要主动说：

```text
实验实现使用 deque.remove 简化 correctness；
生产实现应把队列状态设计成 O(1) 可迁移的数据结构。
```

这是加分点。

---

# 8. Step 6：设计三组严格对照实验

不要只比较“改前 / 改后”。至少跑：

```text
A. upstream baseline
B. chunk cap only
C. chunk cap + fair round-robin
```

固定：

```text
model
prompt set
max_tokens
gpu_memory_utilization
enforce_eager
random seed
```

第一轮建议：

```text
prefill_chunk_size = 256 / 512 / 1024
```

每个 variant 跑 3~5 次，第一次作为 warmup 不计。

汇总：

```text
mean short-request TTFT
p50 short-request TTFT
p95 short-request TTFT
long-request TTFT
mean E2E
completion throughput
```

如果短请求只有 8 个，p95 统计意义很弱；可以把 short_count 增到 32 再测。

---

# 9. Step 7：结果如何判断“优化成功”？

不要提前假设一定全面变快。

可能出现：

```text
short TTFT ↓↓↓
long TTFT ↑
overall throughput ≈ 或轻微 ↓
```

这是完全合理的。

因为你是在做：

```text
latency fairness
而不是纯 throughput maximization
```

真正可以写进简历的是：

```text
针对长短请求混合负载，完成 Chunked Prefill / fairness 调度改造，
并建立 TTFT/E2E/throughput 对照评测。
```

只有当你真的跑出数据后，才能再写：

```text
短请求 p95 TTFT 降低 xx%
```

---

# 10. Step 8：加一个 scheduler-only 单元测试

GPU benchmark 很慢，所以还要写 CPU 级 scheduler correctness test。

目标检查：

```text
fair_prefill=False：partial long request 保持高优先级
fair_prefill=True：partial long request 会移动到尾部
```

可以在 `learning/bench/07_scheduler_policy_test.py` 中用最小 fake Sequence，或者直接构造真实 Sequence + 小 BlockManager。

至少 assert：

```python
assert len(set(seq.block_table)) == len(seq.block_table)
assert seq.num_cached_tokens <= seq.num_tokens
```

并检查 scheduling 轮次：

```text
round 0: long
round 1: short1
round 2: short2
...
```

---

# 11. Step 9：Nsight 是否必须？

今天不是必须。

Scheduler 是 CPU-side policy，最有价值的是：

```text
TTFT distribution + request timeline
```

而不是 GPU kernel profiling。

如果有时间，Day 10 收口时再用 Nsight Systems 看：

```text
长 prefill kernel burst
与 decode kernel 的穿插方式
```

---

# 12. 今天必须形成的 git history

建议：

```bash
git add nanovllm/engine/sequence.py learning/bench/07_scheduler_latency.py
git commit -m "bench: add request latency metrics for scheduler experiments"

git add nanovllm/config.py nanovllm/engine/scheduler.py
git commit -m "feat: add configurable fair chunked-prefill scheduling"
```

不要把所有内容 squash 成一个“update”。面试时 commit history 本身就是证据。

---

# 13. Day 07 面试追问

你要能脱口而出：

### Q1. Continuous Batching 和 Chunked Prefill 区别？

```text
Continuous Batching：请求在 iteration 边界动态加入/退出 batch。
Chunked Prefill：把一个长 prompt 的 prefill 切成多个 iteration，避免单次 prefill 独占 token budget。
```

### Q2. 为什么 chunked prefill 不等于公平？

```text
如果 partial request 每轮仍固定占 waiting 队首，虽然单轮工作量变小，但它仍可能连续优先获得 quota。
```

### Q3. 为什么优化 TTFT 可能伤 throughput？

```text
更细的切分和 request interleaving 可能增加调度、kernel launch、batch shape 波动等开销；这是 latency/throughput trade-off。
```

### Q4. 你的实现是不是 production-ready？

正确回答：

```text
不是。它是用于验证 policy 的小型实验实现；尤其 deque.remove 是 O(n)。但它完整验证了公平调度对 TTFT 的影响，生产实现应重构队列数据结构。
```

---

# 14. 今日验收清单

必须全部满足：

```text
[ ] baseline 未改算法时能跑通
[ ] 能输出 request-level TTFT / E2E
[ ] prefill_chunk_size 可配置
[ ] fair_prefill 可开关
[ ] A/B/C 三组结果使用同一 workload
[ ] 至少一组结果写成 CSV
[ ] 能解释 latency/throughput trade-off
[ ] 没有在简历里虚构性能百分比
```

今天结束后你真正获得的不是“又学了一个名词”，而是：

```text
我读过 Scheduler；
我真的改过 policy；
我知道怎么验证它；
我知道什么结果才算证据。
```


---

# 原 Day08（保持不变）

# `day08-sampler-logit-space-optimization.md`：复现并量化一个真实 Sampler 计算路径优化

> 前置：完成 Day 01~07；理解 logits、temperature、sampling、GPU benchmark。  
> 今日目标：基于 nano-vLLM 当前 Sampler，复现一个“小而硬”的性能优化：利用 Gumbel-Max 等价关系，在 logit space 完成采样，去掉显式 Softmax；做 correctness、microbenchmark、E2E 三层验证。  
> 成果边界：这项思路不是你的原创算法；简历应写“复现并评估 / 优化实现”，不要写“提出”。

---

# 0. 先看原始代码

```bash
sed -n '1,120p' nanovllm/layers/sampler.py
```

当前核心逻辑：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

对于 Qwen3，词表约 15 万维，所以每一个 batch row 都会做一次大 vocabulary Softmax。

今天问：

```text
如果最终只是 argmax，Softmax 的归一化是否真的必要？
```

---

# 1. 今天的目录树

```bash
git switch -c perf/sampler-logit-space
mkdir -p learning/bench learning/results/day08
```

最终：

```text
nano-vllm/
├── nanovllm/layers/sampler.py          # 修改
└── learning/
    ├── bench/
    │   ├── 08_sampler_equivalence.py   # 新增
    │   ├── 08_sampler_microbench.py    # 新增
    │   └── 08_sampler_e2e.py           # 新增
    └── results/day08/
```

---

# 2. Step 1：先推数学，再改代码

当前采样：

```text
z_i = logit_i / T
p_i = exp(z_i) / sum_j exp(z_j)
E_i ~ Exp(1)
output = argmax_i p_i / E_i
```

取 log：

```text
argmax_i p_i / E_i
= argmax_i [log p_i - log E_i]
```

而：

```text
log p_i = z_i - logsumexp(z)
```

`logsumexp(z)` 对所有 token i 相同，因此：

```text
argmax_i [log p_i - log E_i]
= argmax_i [z_i - log E_i]
```

所以可以直接计算：

```text
score_i = logit_i / T - log(E_i)
argmax(score)
```

不再需要：

```text
exp → reduce sum → normalize
```

---

# 3. Step 2：先做完全相同噪声下的 equivalence test

创建：

```text
learning/bench/08_sampler_equivalence.py
```

```python
import torch


def old_sample(logits, temperatures, noise):
    scaled = logits.float() / temperatures[:, None]
    probs = torch.softmax(scaled, dim=-1)
    return (probs / noise.clamp_min(1e-10)).argmax(dim=-1)


def new_sample(logits, temperatures, noise):
    scaled = logits.float() / temperatures[:, None]
    scores = scaled - noise.clamp_min(1e-10).log()
    return scores.argmax(dim=-1)


def main():
    torch.manual_seed(0)
    device = "cuda"
    batch = 32
    vocab = 151936

    logits = torch.randn(batch, vocab, device=device, dtype=torch.float16)
    temperatures = torch.empty(batch, device=device).uniform_(0.3, 1.5)
    noise = torch.empty_like(logits, dtype=torch.float32).exponential_(1)

    a = old_sample(logits, temperatures, noise)
    b = new_sample(logits, temperatures, noise)

    print("equal:", torch.equal(a, b))
    print("mismatch:", (a != b).sum().item())


if __name__ == "__main__":
    main()
```

运行：

```bash
python learning/bench/08_sampler_equivalence.py
```

理想：

```text
equal: True
mismatch: 0
```

如果出现极少量 mismatch，先检查：

```text
float16 overflow / underflow
noise dtype
clamp_min
float conversion位置
```

不要直接宣布“数学错了”。

---

# 4. Step 3：修改 nano-vLLM Sampler

先备份 diff：

```bash
git diff -- nanovllm/layers/sampler.py
```

将核心改为：

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        noise = torch.empty_like(logits).exponential_(1).clamp_min_(1e-10)
        return logits.sub_(noise.log_()).argmax(dim=-1)
```

注意：

```text
这不是“greedy decoding”；
它仍然是随机 sampling，只是用等价的 logit-space 形式实现。
```

---

# 5. Step 4：microbenchmark 要怎么写才可信？

创建：

```text
learning/bench/08_sampler_microbench.py
```

核心原则：

```text
GPU benchmark 必须 warmup + synchronize + 多次迭代
```

完整骨架：

```python
import time
import torch


def old_impl(logits, temperatures):
    x = logits.float() / temperatures[:, None]
    probs = torch.softmax(x, dim=-1)
    return (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


def new_impl(logits, temperatures):
    x = logits.float() / temperatures[:, None]
    noise = torch.empty_like(x).exponential_(1).clamp_min_(1e-10)
    return (x - noise.log()).argmax(dim=-1)


def bench(fn, logits, temperatures, warmup=20, iters=100):
    for _ in range(warmup):
        fn(logits, temperatures)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(logits, temperatures)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def main():
    torch.manual_seed(0)
    for bs in [1, 8, 32, 128]:
        logits = torch.randn(bs, 151936, device="cuda", dtype=torch.float16)
        temperatures = torch.full((bs,), 0.8, device="cuda")
        old_ms = bench(old_impl, logits, temperatures)
        new_ms = bench(new_impl, logits, temperatures)
        print(bs, old_ms, new_ms, old_ms / new_ms)


if __name__ == "__main__":
    main()
```

今天不要只测 batch=1，因为：

```text
batch 越大，vocab-level tensor work 越明显。
```

---

# 6. Step 5：为什么 microbenchmark 变快，不代表 E2E 一定明显变快？

LLM decode 单步包含：

```text
Attention
MLP
LM head
Sampler
Scheduler bookkeeping
```

Sampler 只占其中一部分。

因此很可能出现：

```text
Sampler kernel: 1.3x / 1.8x 更快
E2E tokens/s: 只提升 1%~几%
```

这完全合理。

面试时最差的回答是：

```text
microbenchmark 快 2x，所以整个模型快 2x。
```

今天必须同时测 E2E。

---

# 7. Step 6：E2E benchmark

创建：

```text
learning/bench/08_sampler_e2e.py
```

固定：

```text
同一个模型 Qwen3-0.6B
同一组 prompts
同一个 temperature
同样 max_tokens
同样 eager/cudagraph 配置
```

推荐：

```text
prompts = 64 或 128
prompt length ≈ 128
max_tokens = 128
```

分别在：

```text
commit A: original sampler
commit B: logit-space sampler
```

跑 3~5 次。

记录：

```text
wall_time
completion_tokens
completion_tokens/s
peak GPU memory
```

最干净的办法不是写复杂 toggle，而是用 git commit 切换：

```bash
git checkout <baseline_commit>
python learning/bench/08_sampler_e2e.py

git checkout <optimized_commit>
python learning/bench/08_sampler_e2e.py
```

---

# 8. Step 7：可选 Nsight / torch profiler

如果时间够，可进一步问：

```text
Softmax 消失后，kernel launch 数有没有减少？
GPU time 是否从 softmax 转移到 exponential/log/argmax？
```

但 Day 08 的必修不是 Nsight，而是：

```text
数学正确性 + microbenchmark + E2E benchmark
```

---

# 9. Step 8：记录结果模板

保存：

```text
learning/results/day08/microbench.csv
```

```csv
batch,vocab,old_ms,new_ms,speedup
1,151936,...,...,...
8,151936,...,...,...
32,151936,...,...,...
128,151936,...,...,...
```

E2E：

```csv
variant,requests,prompt_tokens,max_tokens,wall_s,completion_tokens,tokens_per_s
baseline,64,128,128,...,...,...
logit_space,64,128,128,...,...,...
```

---

# 10. Step 9：git commit

```bash
git add learning/bench/08_sampler_equivalence.py learning/bench/08_sampler_microbench.py
git commit -m "bench: validate logit-space sampling equivalence"

git add nanovllm/layers/sampler.py learning/bench/08_sampler_e2e.py
git commit -m "perf: remove explicit softmax from sampler path"
```

---

# 11. 今天在简历里能写什么？

推荐：

```text
复现并评估 Sampler 计算路径优化，基于 Gumbel-Max 等价变换去除显式 Softmax，
完成 correctness、GPU microbenchmark 与端到端性能对照。
```

不要写：

```text
提出全新采样算法
```

也不要在没有真实数字前写：

```text
吞吐提升 xx%
```

---

# 12. 面试追问

### Q1. 为什么去 Softmax 后分布没变？

因为 `log softmax(z)` 与 `z` 只差一个对所有 token 相同的 `-logsumexp(z)`，不会改变加入 Gumbel noise 后的 argmax。

### Q2. 为什么用了 exponential noise？

`-log(Exp(1))` 与 Gumbel-Max trick 的噪声形式等价，可以从 categorical distribution 采样。

### Q3. 为什么 E2E 提升可能很小？

Sampler 不是 decode 的唯一热点；模型 matmul、attention 和 LM head 通常占更大比例。

### Q4. 你做的是原创优化吗？

正确回答：

```text
不是。我在 nano-vLLM 当前实现上复现并验证了这一等价优化，重点是源码改造、正确性证明和系统 benchmark。
```

---

# 13. Day 08 验收

```text
[ ] 同一 noise 下 old/new token 完全一致或能解释极少数浮点边界差异
[ ] 真实修改 nanovllm/layers/sampler.py
[ ] microbenchmark 至少覆盖 4 个 batch size
[ ] E2E 使用同一 workload 对照
[ ] 所有数据保存 CSV
[ ] 能解释为什么 microbenchmark ≠ E2E speedup
[ ] 简历中不把该思路包装成原创算法
```


---

# 原 Day09（保持不变）

# `day09-qwen3-speculative-decoding-prototype.md`：Qwen3-0.6B → Qwen3-4B 的 Greedy Speculative Decoding 原型

> 前置：完成 Day 01~08，尤其是 Prefill/Decode、KV Cache、Sampler、benchmark。  
> 今日目标：在 RTX 5090 24GB 上真正跑通一个**正确、可测、边界清晰**的 Greedy Speculative Decoding 原型，并做 K sweep。  
> 模型固定：`Qwen/Qwen3-0.6B-Base` 作为 draft，`Qwen/Qwen3-4B-Base` 作为 target，BF16。两者属于同一 Qwen3 dense family，词表均为 151936，适合直接做 token-level proposal/verification。  
> 今日成果边界：**独立原型，不声称已经把 Speculative Decoding 完整集成进 nano-vLLM Scheduler / Paged KV Cache / CUDA Graph。** 最后会把 integration points 写清楚。

---

# 0. 为什么最终选 0.6B → 4B，而不是更小或更大？

你只有 10 天，第一目标是：

```text
一定跑通
+
足够真实
+
有 benchmark 厚度
```

三种候选：

```text
0.6B -> 1.7B：最轻，但 target/draft 差距较小
0.6B -> 4B：显存宽松，compute gap 足够明显，最平衡
0.6B -> 8B：可尝试，但 24GB 下两模型 + KV + runtime 余量明显更紧
```

BF16 权重粗估：

```text
0.6B ≈ 1.2 GB
4B   ≈ 8 GB
总权重 ≈ 9.2 GB
```

24GB 显存有足够空间留给：

```text
两套 KV cache
activations
CUDA workspace
PyTorch allocator
```

所以主实验固定 0.6B → 4B。

---

# 1. 为什么先用 Base 模型？

今天重点是算法本身：

```text
draft proposal
→ target parallel verification
→ longest matching prefix
→ accept/reject
```

Base 模型可以减少：

```text
chat template
thinking mode
system prompt
特殊 generation config
```

带来的干扰。

等原型跑通后，如果还有时间，再换 Instruct：

```text
Qwen3-0.6B
Qwen3-4B
```

做更贴近真实聊天的测试。

---

# 2. 今日目录树

```bash
git switch -c feat/greedy-spec-decode-prototype
mkdir -p learning/speculative learning/bench learning/results/day09
```

最终新增：

```text
nano-vllm/
└── learning/
    ├── speculative/
    │   ├── greedy_spec_decode.py
    │   └── NANOVLLM-INTEGRATION.md
    ├── bench/
    │   └── 09_speculative_bench.py
    └── results/day09/
        ├── correctness.txt
        └── k_sweep.csv
```

Day 09 第一版**不修改 `nanovllm/` 主代码**。这是刻意的风险控制：先把算法和 benchmark 做正确，再谈 engine integration。

---

# 3. Step 1：安装/下载模型

如果模型还没下载：

```bash
huggingface-cli download Qwen/Qwen3-0.6B-Base \
  --local-dir ~/huggingface/Qwen3-0.6B-Base

huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir ~/huggingface/Qwen3-4B-Base
```

如果你的 CLI 是新版本：

```bash
hf download Qwen/Qwen3-0.6B-Base \
  --local-dir ~/huggingface/Qwen3-0.6B-Base

hf download Qwen/Qwen3-4B-Base \
  --local-dir ~/huggingface/Qwen3-4B-Base
```

检查磁盘：

```bash
du -sh ~/huggingface/Qwen3-0.6B-Base ~/huggingface/Qwen3-4B-Base
```

---

# 4. Step 2：验证 tokenizer / vocab 完全兼容

创建一个临时检查：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig

D = "~/huggingface/Qwen3-0.6B-Base"
T = "~/huggingface/Qwen3-4B-Base"
import os
D, T = os.path.expanduser(D), os.path.expanduser(T)

dc = AutoConfig.from_pretrained(D)
tc = AutoConfig.from_pretrained(T)
dt = AutoTokenizer.from_pretrained(D)
tt = AutoTokenizer.from_pretrained(T)

print("draft arch:", dc.architectures)
print("target arch:", tc.architectures)
print("draft vocab:", dc.vocab_size)
print("target vocab:", tc.vocab_size)
print("tokenizer vocab sizes:", len(dt), len(tt))

s = "Speculative decoding verifies draft tokens in parallel."
print(dt.encode(s) == tt.encode(s))
PY
```

必须至少确认：

```text
Qwen3ForCausalLM
vocab_size = 151936
同一字符串 encode 结果一致
```

如果 tokenizer 不一致，不要继续。

---

# 5. Step 3：先写 target-only greedy baseline

创建：

```text
learning/speculative/greedy_spec_decode.py
```

先写模型加载：

```python
from __future__ import annotations

import time
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DRAFT_PATH = "~/huggingface/Qwen3-0.6B-Base"
TARGET_PATH = "~/huggingface/Qwen3-4B-Base"


@dataclass
class DecodeStats:
    output_tokens: int = 0
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    target_forward_calls: int = 0
    draft_forward_calls: int = 0
    wall_s: float = 0.0

    @property
    def acceptance_rate(self):
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.drafted_tokens

    @property
    def tokens_per_target_call(self):
        if self.target_forward_calls == 0:
            return 0.0
        return self.output_tokens / self.target_forward_calls
```

加载模型：

```python
import os


def load_models():
    draft_path = os.path.expanduser(DRAFT_PATH)
    target_path = os.path.expanduser(TARGET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(target_path)

    draft = AutoModelForCausalLM.from_pretrained(
        draft_path,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    return tokenizer, draft, target
```

第一次加载后：

```python
print(torch.cuda.memory_summary(abbreviated=True))
```

确保没有接近 OOM。

---

# 6. Step 4：先写“最笨但正确”的 greedy baseline

第一版 target-only 每生成一个 token，都允许直接全 prefix forward；效率低，但适合验证最终 token sequence。

```python
@torch.inference_mode()
def target_greedy_reference(model, input_ids, max_new_tokens):
    ids = input_ids.clone()
    stats = DecodeStats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
        stats.target_forward_calls += 1
        stats.output_tokens += 1

    torch.cuda.synchronize()
    stats.wall_s = time.perf_counter() - t0
    return ids, stats
```

为什么先写这个？

```text
它不是性能 baseline，
它是 correctness oracle。
```

后面的 speculative output 在 greedy 条件下必须与 target-only greedy sequence 一致。

---

# 7. Step 5：理解 verification 的位置对齐

假设 prefix 长度为 L，draft 提议：

```text
d1 d2 d3 d4
```

把：

```text
prefix + d1 + d2 + d3 + d4
```

送入 target。

Causal LM 的 `logits[:, p, :]` 预测的是：

```text
位置 p+1 的 token
```

因此：

```text
预测 d1：看 prefix 最后一个位置的 logits
预测 d2：看 d1 位置的 logits
预测 d3：看 d2 位置的 logits
...
```

这是 Day 09 最容易写错的 off-by-one。

你必须画：

```text
input:   x0 x1 ... x(L-1) d1 d2 d3
logits:  ->x1 ... ->d1     ->d2 ->d3 ->extra
```

---

# 8. Step 6：实现 draft K-token proposal

第一版仍然允许使用 HF `past_key_values`，这样 draft 不用每步重算整个 prefix。

伪代码：

```python
@torch.inference_mode()
def draft_k_tokens(draft, prefix_ids, k):
    outputs = draft(input_ids=prefix_ids, use_cache=True)
    past = outputs.past_key_values
    token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    proposals = [token]

    for _ in range(k - 1):
        outputs = draft(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        proposals.append(token)

    return torch.cat(proposals, dim=1)
```

这版的一个不足：

```text
每个 speculative round 又重新 prefill draft prefix。
```

稍后我们再做 draft KV reuse。

---

# 9. Step 7：实现一次 target parallel verification

```python
@torch.inference_mode()
def verify_once(target, prefix_ids, draft_ids):
    """
    prefix_ids: [1, L]
    draft_ids:  [1, K]

    return:
      target_tokens: [1, K+1]
      其中前 K 个用于对比 draft，最后 1 个用于 all-accepted 时 bonus token
    """
    L = prefix_ids.size(1)
    K = draft_ids.size(1)

    joined = torch.cat([prefix_ids, draft_ids], dim=1)
    logits = target(input_ids=joined).logits

    # logits position L-1 predicts draft token #1
    # ...
    # logits position L+K-1 predicts bonus token
    verify_logits = logits[:, L - 1 : L + K, :]
    target_tokens = verify_logits.argmax(dim=-1)
    return target_tokens
```

检查 shape：

```text
target_tokens.shape == [1, K+1]
```

---

# 10. Step 8：实现 longest-prefix accept/reject

```python
def accept_greedy(draft_ids, target_tokens):
    K = draft_ids.size(1)
    accepted = []

    for i in range(K):
        d = draft_ids[0, i].item()
        t = target_tokens[0, i].item()

        if d == t:
            accepted.append(d)
            continue

        # first mismatch: output target token and stop
        accepted.append(t)
        return accepted, i, False

    # all K draft tokens matched, append one bonus target token
    bonus = target_tokens[0, K].item()
    accepted.append(bonus)
    return accepted, K, True
```

返回的 `i` 是：

```text
accepted_draft_tokens
```

而 `accepted` 总长度：

```text
mismatch 情况 = i + 1
all accepted = K + 1
```

---

# 11. Step 9：拼成最小 Speculative Decode loop

```python
@torch.inference_mode()
def speculative_greedy(draft, target, input_ids, max_new_tokens, k):
    ids = input_ids.clone()
    stats = DecodeStats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while stats.output_tokens < max_new_tokens:
        remain = max_new_tokens - stats.output_tokens
        cur_k = min(k, remain)

        proposals = draft_k_tokens(draft, ids, cur_k)
        stats.draft_forward_calls += cur_k
        stats.drafted_tokens += cur_k

        target_tokens = verify_once(target, ids, proposals)
        stats.target_forward_calls += 1

        emitted, accepted_draft, all_accepted = accept_greedy(
            proposals, target_tokens
        )

        emitted = emitted[:remain]
        stats.accepted_draft_tokens += min(accepted_draft, len(emitted))
        stats.output_tokens += len(emitted)

        new_tokens = torch.tensor(
            emitted,
            dtype=ids.dtype,
            device=ids.device,
        ).unsqueeze(0)
        ids = torch.cat([ids, new_tokens], dim=1)

    torch.cuda.synchronize()
    stats.wall_s = time.perf_counter() - t0
    return ids, stats
```

第一版跑通即可。

注意：这版 target 每 round 仍然重算整个 prefix，所以**不能拿 wall-clock speedup 当最终 SpecDecode 性能结论**。它主要用于：

```text
算法正确性
acceptance metrics
K sweep
```

---

# 12. Step 10：第一条硬验收——结果必须与 target greedy 一致

加入：

```python
prompt = "Speculative decoding is useful because"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

ref_ids, ref_stats = target_greedy_reference(
    target, inputs, max_new_tokens=64
)

spec_ids, spec_stats = speculative_greedy(
    draft, target, inputs, max_new_tokens=64, k=4
)

print("same token sequence:", torch.equal(ref_ids, spec_ids))
print("acceptance_rate:", spec_stats.acceptance_rate)
print("tokens/target_call:", spec_stats.tokens_per_target_call)
```

必须优先追求：

```text
same token sequence: True
```

如果 False，先检查：

```text
verification logits slice
bonus token位置
first mismatch 行为
max_new_tokens 截断
```

不要先看性能。

---

# 13. Step 11：做 K sweep

创建：

```text
learning/bench/09_speculative_bench.py
```

固定一组 prompts，例如：

```python
PROMPTS = [
    "The main purpose of virtual memory is",
    "FlashAttention improves attention efficiency by",
    "A compiler intermediate representation is",
    "Speculative decoding can reduce inference latency when",
    "Paged KV cache is useful because",
    "CUDA Graph reduces overhead by",
    "Continuous batching differs from static batching because",
    "The role of a scheduler in an LLM serving engine is",
]
```

K：

```text
1, 2, 4, 6, 8
```

每组记录：

```text
K
drafted_tokens
accepted_draft_tokens
acceptance_rate
output_tokens
target_forward_calls
tokens_per_target_call
wall_s  # 只作为原型参考，不宣传为最终加速数字
```

CSV：

```csv
k,drafted,accepted,acceptance_rate,output_tokens,target_calls,tokens_per_target_call,wall_s
1,...
2,...
4,...
6,...
8,...
```

你要观察的不是“K 越大越好”，而是：

```text
K ↑
潜在一次确认 token 数 ↑
但 draft 成本 ↑
且后段 token 更容易 mismatch
```

---

# 14. Step 12：做第二版——至少复用 draft KV

第一版每轮 `draft_k_tokens()` 都重新算整个 prefix，不够像真实系统。

今天如果还有 1~2 小时，把 draft state 持久化：

```text
draft_past_key_values
+
当前已经提交的 token
```

思路：

```text
首次：prefill 全 prefix，得到 draft past
之后每轮：
只把新 committed target tokens 喂给 draft，更新 past
再继续 autoregressive draft K tokens
```

难点在 rejection：

```text
本轮 draft 的后缀可能被 target reject，
因此不要直接把所有 proposal 的 draft KV 永久提交。
```

最简单风险控制：

```text
每 round 的 speculative proposal 使用临时 draft past；
确认 emitted token 后，再从 committed past 上推进 emitted token。
```

这不是最优，但 correctness 清晰。

完成后你可以在 README 中写：

```text
prototype supports committed-prefix KV reuse for draft model
```

但不要写“完成 Paged KV rollback”。

---

# 15. Step 13：为什么今天不直接把它塞进 nano-vLLM ModelRunner？

因为真正 integration 至少要改：

```text
Sequence
    - 一轮可能 commit >1 completion token
    - 需要 speculative state

Scheduler
    - token budget 不再是 decode 每 seq 固定 1 token
    - speculative round 如何计费

BlockManager
    - 为 K 个 candidate token 预留 slot
    - rejection 后如何处理未提交 slots

ModelRunner
    - target 一次返回 K+1 positions 的 logits
    - 不能 run() 后立刻只 sample 一个 token

KV Cache
    - proposal KV / verified KV 的 commit 边界
    - rejection rollback / logical invalidation

CUDA Graph
    - decode shape 从 batch×1 变成 batch×K
```

这些不是一天能可靠改完的。

所以 Day 09 的正确项目描述是：

```text
实现 Greedy Speculative Decoding 独立原型，并完成 acceptance / accepted length / target-call reduction 分析；
分析接入 nano-vLLM Paged KV/Scheduler 时的 commit/rollback 改造点。
```

而不是：

```text
在 nano-vLLM 中完整实现 production Speculative Decoding。
```

---

# 16. Step 14：写 `NANOVLLM-INTEGRATION.md`

创建：

```text
learning/speculative/NANOVLLM-INTEGRATION.md
```

必须画这张状态机：

```text
RUNNING Sequence
      │
      ├─ draft K tokens
      │
      ├─ reserve K (+1) logical slots
      │
      ├─ target verify
      │
      ├─ accept m draft tokens
      │
      ├─ commit accepted prefix
      │
      ├─ on mismatch: commit target correction token
      │
      └─ invalidate / recycle rejected speculative suffix
```

然后逐文件列 integration point：

```text
nanovllm/engine/sequence.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/model_runner.py
nanovllm/layers/attention.py
```

这样面试官问：

```text
“为什么没直接集成？”
```

你可以回答：

```text
因为真正难点不是 accept/reject 算法，而是 multi-token commit 与 Paged KV 生命周期；
我先把算法闭环跑通，再把需要改动的 engine state 明确拆分。
```

---

# 17. Step 15：5090 24GB 的显存安全策略

主实验：

```text
Draft  = Qwen3-0.6B-Base BF16
Target = Qwen3-4B-Base BF16
batch  = 1
max prompt length 先 512~1024
max_new_tokens 64~128
K <= 8
```

先执行：

```bash
watch -n 1 nvidia-smi
```

如果显存峰值异常：

```text
1. 先减 prompt length
2. 不要开大 batch
3. 删除不用的 logits / outputs 引用
4. torch.cuda.empty_cache() 只在实验阶段边界使用
```

不要一上来做 8B target。

如果 0.6B→4B 全部完成且还有时间，Day 10 可以**额外**尝试：

```text
Qwen3-0.6B -> Qwen3-8B
```

作为 scaling experiment；它不是必修，也不要写进简历主成果。

---

# 18. Step 16：git commit

```bash
git add learning/speculative/greedy_spec_decode.py
git commit -m "feat: add greedy speculative decoding prototype for Qwen3"

git add learning/bench/09_speculative_bench.py learning/speculative/NANOVLLM-INTEGRATION.md
git commit -m "bench: add speculative decoding K-sweep and integration analysis"
```

---

# 19. Day 09 面试追问

### Q1. 为什么 draft/target tokenizer 必须兼容？

proposal 是 token id；target 需要直接验证相同 token id 的语义。如果 vocab/tokenizer 不一致，就不能简单逐 token 对比。

### Q2. 为什么 0.6B→4B 比 0.6B→1.7B 更适合性能实验？

SpecDecode 的价值依赖 target 显著更昂贵；1.7B 与 0.6B 的 compute gap 较小，draft overhead 更容易吃掉收益。4B 在 24GB 上仍有宽裕显存，又有更明显的成本差。

### Q3. `acceptance_rate` 和 `tokens_per_target_call` 区别？

```text
acceptance_rate = accepted draft tokens / drafted tokens

tokens_per_target_call = 最终输出 token / target verification 次数
```

真正和减少 target serial calls 更直接相关的是后者。

### Q4. 为什么原型 wall-clock 不一定比 baseline 快？

第一版为了 correctness 仍会重复计算 target prefix，且 HF Python overhead 很大；它验证的是 algorithmic target-call reduction，而不是 production optimized speedup。

### Q5. production 集成最难的是什么？

```text
Paged KV 下多 token reserve / commit / rejection rollback，
以及 Scheduler 对 speculative token budget 的 accounting。
```

---

# 20. Day 09 验收清单

```text
[ ] Qwen3-0.6B-Base + Qwen3-4B-Base 同时加载到 5090 24GB
[ ] tokenizer/vocab compatibility 已验证
[ ] target-only greedy reference 跑通
[ ] speculative output 与 greedy target token sequence 一致
[ ] K=1/2/4/6/8 sweep 跑通
[ ] 记录 acceptance_rate
[ ] 记录 tokens_per_target_call
[ ] 至少实现/分析 draft KV reuse
[ ] 写 NANOVLLM-INTEGRATION.md
[ ] 明确写出“未完成 production engine integration”
```

完成这一天后，你可以非常稳妥地说：

```text
我不是只看过 Speculative Decoding 论文；
我用真实 Qwen3 draft/target 模型从零跑通了 greedy proposal/parallel verification/accept-reject，
验证了 token-level correctness，做了 K sweep，并能解释如何接入 Paged KV engine。
```
