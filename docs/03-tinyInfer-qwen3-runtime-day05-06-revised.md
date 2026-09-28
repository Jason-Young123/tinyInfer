# tinyInfer 新版讲义 03：Day 05–06 —— 在 Mixed Runtime 上接入真实 Qwen3，并建立 Correctness / Benchmark 工程闭环

> 本篇覆盖并替代原 `03-tinyInfer-qwen3-runtime-day05-09.md` 的 Day05–06 部分。
>
> **暂不包含 Day07–09。** 后续 scheduler optimization、sampler optimization、speculative decoding 应在真实模型 baseline 稳定后重新设计。
>
> 前置要求：已经完成新版 `02-A-tinyInfer-runtime-consolidation.md`。此时 runtime 必须已经支持：
>
> - paged physical KV block table；
> - persistent Prefix Cache + LRU eviction；
> - chunked prefill；
> - decode + prefill mixed batching；
> - unified `SchedulerOutput`；
> - unified `ModelRunner.prepare_batch()`；
> - cached-prefix aware educational PagedAttention；
> - CPU pytest 回归。

---

# Day 05：从 Toy Runtime 到真正 Qwen3 Forward

# 0. Day05 的原则：这一天只替换“模型执行面”，不要再改调度语义

02-A 结束时：

```text
请求生命周期 / token budget / KV block / mixed metadata
```

应该都已经能独立测试。

Day05 的工作是把：

```text
ToyModelRunner
```

逐层替换成：

```text
Token Embedding
  ↓
Qwen3 Decoder Layers
  ├── RMSNorm
  ├── QKV Projection
  ├── RoPE
  ├── Paged Attention
  ├── O Projection
  ├── RMSNorm
  └── SwiGLU MLP
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
Sampler
```

**不要在今天同时引入 CUDA Graph、Triton kernel、TP、多 GPU。**

先做单卡 eager correctness。

---

# 1. Day05 最终目录树

新增：

```text
tinyinfer/
├── layers/
│   ├── __init__.py
│   ├── activation.py
│   ├── attention.py              # 02-A 版本继续保留
│   ├── embed_head.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── rotary_embedding.py
│   └── sampler.py
├── models/
│   ├── __init__.py
│   └── qwen3.py
└── utils/
    ├── __init__.py
    ├── context.py
    ├── debug.py
    └── loader.py
```

修改：

```text
tinyinfer/config.py
tinyinfer/engine/model_runner.py
tinyinfer/engine/llm_engine.py
tinyinfer/llm.py
```

新增测试：

```text
tests/
├── test_linear.py
├── test_rope.py
├── test_qwen3_shapes.py
├── test_loader.py
└── test_hf_equivalence.py
```

---

# 2. Step 1：先让 Config 真正读取 Hugging Face config

修改：

```text
tinyinfer/config.py
```

当前 `hf_config` 只是一个占位字段。

建议增加：

```python
from transformers import AutoConfig


def load_hf_config(self):
    if self.model is None:
        raise ValueError("model path/name is required for real runtime")

    self.hf_config = AutoConfig.from_pretrained(
        self.model,
        trust_remote_code=True,
    )

    if self.max_model_len > self.hf_config.max_position_embeddings:
        self.max_model_len = self.hf_config.max_position_embeddings
```

在真正 runtime 初始化时调用，而不是 import 时调用。

你至少要能打印：

```python
cfg = Config(model="Qwen/Qwen3-0.6B")
cfg.load_hf_config()

print(cfg.hf_config.hidden_size)
print(cfg.hf_config.num_hidden_layers)
print(cfg.hf_config.num_attention_heads)
print(cfg.hf_config.num_key_value_heads)
print(cfg.hf_config.head_dim)
```

## pytest

如果测试环境不希望联网，就让它只接本地 model path；HF 集成测试用 marker：

```toml
[tool.pytest.ini_options]
markers = [
    "model: requires a local Hugging Face model checkpoint",
]
```

---

# 3. Step 2：先写最普通的 Linear，不急着 TP

创建：

```text
tinyinfer/layers/linear.py
```

```python
import torch
import torch.nn.functional as F
from torch import nn


class LinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=dtype,
                device=device,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

## 为什么先不 TP？

你现在真正要验证的是：

```text
权重名称是否对上
QKV shape 是否对上
RoPE 是否对上
KV cache 是否对上
logits 是否对上
```

TP 会同时引入：

```text
shard shape
rank
collective
packed weight slicing
```

会让 correctness debug 难度急剧上升。

先单卡；Day06 baseline 稳定后再加 TP 是更合理的学习顺序。

---

# 4. Step 3：实现 packed QKV projection

Qwen3 attention 通常有：

```text
Q: num_heads × head_dim
K: num_kv_heads × head_dim
V: num_kv_heads × head_dim
```

因此三个输出维度不一定相同。

在 `linear.py` 增加：

```python
class QKVParallelLinear(LinearBase):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        bias=False,
        dtype=None,
        device=None,
    ):
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        out_features = self.q_size + 2 * self.kv_size

        super().__init__(
            hidden_size,
            out_features,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def split_qkv(self, x):
        q, k, v = torch.split(
            x,
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        return q, k, v
```

注意：名字里暂时保留 `Parallel` 不代表你已经真的 TP；它只是为后续升级保持接口接近常见推理框架。

---

# 5. Step 4：实现 Gate-Up packed projection

Qwen3 MLP 使用 SwiGLU：

```text
gate = W_gate x
up   = W_up x
out  = W_down (SiLU(gate) * up)
```

可以把 gate/up 合并：

```python
class MergedGateUpLinear(LinearBase):
    def __init__(self, hidden_size, intermediate_size, **kwargs):
        self.intermediate_size = intermediate_size
        super().__init__(
            hidden_size,
            2 * intermediate_size,
            **kwargs,
        )

    def split_gate_up(self, x):
        return x.chunk(2, dim=-1)
```

---

# 6. Step 5：RMSNorm

创建：

```text
tinyinfer/layers/layernorm.py
```

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(input_dtype)
```

为什么先显式转 FP32？

为了 correctness baseline 更稳，优化可以后做。

---

# 7. Step 6：SwiGLU

创建：

```text
tinyinfer/layers/activation.py
```

```python
class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
```

测试 shape：

```python
x = torch.randn(7, 2 * 16)
y = SiluAndMul()(x)
assert y.shape == (7, 16)
```

---

# 8. Step 7：实现 RoPE，但一定要让 position 来自 mixed batch metadata

创建：

```text
tinyinfer/layers/rotary_embedding.py
```

最重要的接口不是具体优化方式，而是：

```python
forward(q, k, positions)
```

其中：

```text
positions
```

来自 02-A 的：

```python
ModelRunner.prepare_batch()
```

例如一个 mixed batch：

```text
seq0 decode position 12
seq1 prefill positions 4,5,6,7
seq2 decode position 29
```

flat：

```python
positions = [12, 4, 5, 6, 7, 29]
```

RoPE 不应该关心这些 token 来自 prefill 还是 decode。

## 教学实现

你可以先预计算：

```python
inv_freq = 1.0 / (
    base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        / rotary_dim
    )
)
```

然后：

```python
freqs = torch.outer(positions.float(), inv_freq)
cos = freqs.cos()
sin = freqs.sin()
```

按照 Qwen3/HF 的 rotary layout 对 q/k 做旋转。

这里建议直接写 `tests/test_rope.py` 与 Hugging Face rotary helper 做小 tensor 对齐，而不要凭肉眼判断。

---

# 9. Step 8：Embedding 与 LM Head

创建：

```text
tinyinfer/layers/embed_head.py
```

第一版只做普通 embedding：

```python
class VocabEmbedding(nn.Embedding):
    pass
```

LM head：

```python
class LMHead(nn.Linear):
    def __init__(self, hidden_size, vocab_size, bias=False, **kwargs):
        super().__init__(hidden_size, vocab_size, bias=bias, **kwargs)
```

如果 Qwen3 checkpoint 使用 tied embeddings，loader 时要处理：

```text
lm_head.weight
```

是否独立存在。

不要在这里先做 vocab TP。

---

# 10. Step 9：Sampler 先做正确，不先做花式优化

创建：

```text
tinyinfer/layers/sampler.py
```

你的 `SamplingParams` 目前只允许：

```text
temperature > 0
```

因此可以：

```python
class Sampler(nn.Module):
    def forward(self, logits, temperatures):
        scaled = logits / temperatures.unsqueeze(-1)
        probs = torch.softmax(scaled.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

但是 correctness test 时建议先增加 deterministic greedy 模式。

修改 `SamplingParams`：

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    greedy: bool = False
```

Sampler：

```python
if greedy:
    return logits.argmax(dim=-1)
```

为什么值得加？

因为和 HF 对齐时：

```text
随机 sampling 会把模型差异与 RNG 差异混在一起
```

Greedy token-by-token equivalence 是最简单的硬验收。

---

# 11. Step 10：组装 Qwen3 Attention

创建：

```text
tinyinfer/models/qwen3.py
```

Attention 架构：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx, kv_cache):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = LinearBase(...)
        self.rotary_emb = RotaryEmbedding(...)

        self.attn = Attention(
            layer_idx=layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim ** -0.5,
            kv_cache=kv_cache,
            block_size=kv_cache.block_size,
        )
```

forward：

```python
def forward(self, hidden_states, positions):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = self.qkv_proj.split_qkv(qkv)

    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)

    q, k = self.rotary_emb(q, k, positions)
    out = self.attn(q, k, v)

    out = out.reshape(-1, self.num_heads * self.head_dim)
    return self.o_proj(out)
```

注意：这里完全没有：

```python
if is_prefill:
```

因为 mixed semantics 已经由 02-A 的 Attention/context 消化掉。

---

# 12. Step 11：组装 MLP

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedGateUpLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.act = SiluAndMul()
        self.down_proj = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act(x)
        return self.down_proj(x)
```

---

# 13. Step 12：组装 Decoder Layer

Qwen-style pre-norm：

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, kv_cache):
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = Qwen3Attention(config, layer_idx, kv_cache)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states, positions):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

注意：具体 Qwen3 版本如果有 q/k norm 等结构，必须以你本地 checkpoint 的 `config.json` 与 HF 对应实现为准；不要强行把 Qwen2 架构当作 Qwen3。

---

# 14. Step 13：组装整个模型

```python
class Qwen3Model(nn.Module):
    def __init__(self, config, kv_cache):
        super().__init__()
        self.embed_tokens = VocabEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i, kv_cache)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.norm(hidden_states)
```

然后：

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config, kv_cache):
        super().__init__()
        self.model = Qwen3Model(config, kv_cache)
        self.lm_head = LMHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

这里故意让：

```text
forward → hidden states
compute_logits → 只对需要 sampling 的位置做
```

因为 mixed batch 中 partial prefill token 根本不需要 LM head。

---

# 15. Step 14：ModelRunner 要只对 sample positions 计算 logits

02-A 已经有：

```text
SchedulerOutput.items
item.sample_after
```

因此 `ModelRunner.run()`：

```python
def run(self, output: SchedulerOutput):
    try:
        input_ids, positions = self.prepare_batch(output)
        hidden_states = self.model(input_ids, positions)

        sample_flat_indices = []
        sample_items = []
        cursor = 0

        for item in output.items:
            cursor += item.num_tokens
            if item.sample_after:
                sample_flat_indices.append(cursor - 1)
                sample_items.append(item)

        if not sample_items:
            return {}

        idx = torch.tensor(
            sample_flat_indices,
            dtype=torch.long,
            device=hidden_states.device,
        )
        sample_hidden = hidden_states.index_select(0, idx)
        logits = self.model.compute_logits(sample_hidden)

        tokens = self.sampler(...)

        return {
            item.seq.seq_id: int(token)
            for item, token in zip(sample_items, tokens.tolist())
        }
    finally:
        reset_context()
```

这样一个 100-token partial prefill chunk 不会做：

```text
100 × vocab_size
```

的无用 LM-head projection。

---

# 16. Step 15：写 loader 前，先理解 checkpoint name mapping

创建：

```text
tinyinfer/utils/loader.py
```

你会遇到两类权重：

```text
A. 1:1 权重
model.layers.0.self_attn.o_proj.weight
→ 同名直接 copy

B. packed 权重
HF:
  q_proj.weight
  k_proj.weight
  v_proj.weight

tinyInfer:
  qkv_proj.weight
```

所以 loader 不可能永远：

```python
state_dict[name].copy_(tensor)
```

你需要显式 packed mapping。

---

# 17. Step 16：实现最小 Safetensors loader

基础流程：

```python
from pathlib import Path
from safetensors import safe_open


def iter_safetensor_weights(model_path):
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
```

然后构造参数字典：

```python
params = dict(model.named_parameters())
```

对于普通权重：

```python
params[target_name].data.copy_(tensor)
```

对于 Q/K/V：

```text
q → qkv weight [0:q_size]
k → qkv weight [q_size:q_size+kv_size]
v → qkv weight [...]
```

Gate/Up 同理。

---

# 18. Step 17：Loader 必须做 strict accounting

加载完后至少保留：

```python
loaded_params = set()
unknown_checkpoint_weights = []
```

最后：

```python
missing = set(params) - loaded_params
```

开发阶段直接：

```python
if missing:
    raise RuntimeError(f"unloaded model parameters: {sorted(missing)[:20]}")
```

为什么？

因为“模型能 forward”并不代表“模型加载正确”。

一个漏加载的 parameter 仍然可能是随机值，然后输出看起来只是“有点奇怪”。

这类 bug 非常难查。

---

# 19. Step 18：KV Cache capacity 不能在 Config 默认值上硬编码

进入真实模型后：

```text
num_kvcache_blocks
```

必须按显存算。

每个 physical block 的 KV bytes：

```text
num_layers
× 2                       # K + V
× block_size
× num_kv_heads
× head_dim
× dtype_bytes
```

所以：

```python
bytes_per_block = (
    num_layers
    * 2
    * block_size
    * num_kv_heads
    * head_dim
    * dtype_bytes
)
```

可用显存：

```python
total = torch.cuda.get_device_properties(device).total_memory
usable = int(total * gpu_memory_utilization)
```

但不能直接：

```python
num_blocks = usable // bytes_per_block
```

因为模型参数、CUDA context、临时 activation 已经占显存。

---

# 20. Step 19：正确的初始化顺序

建议 `ModelRunner.__init__()`：

```text
1. 加载 HF config
2. 构造模型（暂时没有真实 KV storage）
3. 加载模型权重到 GPU
4. 做一个小 warmup / measure peak model working memory
5. 获取当前 allocated/reserved memory
6. 按 gpu_memory_utilization 剩余额度算 num_kvcache_blocks
7. 创建 PagedKVCache
8. 把同一个 KV cache object 注入每层 Attention
9. 再执行正式 warmup
```

为了避免“先构造 Attention 就必须有 kv_cache”的循环依赖，你可以选择两种教学方案。

### 方案 A：先算理论容量，再创建模型

优点：简单。

缺点：model working memory 估算不准。

### 方案 B：Attention 允许后绑定 KV cache

推荐。

`Attention.__init__()`：

```python
self.kv_cache = None
```

新增：

```python
def set_kv_cache(self, kv_cache):
    self.kv_cache = kv_cache
```

模型加载 + measure 后创建 cache，再遍历：

```python
for module in model.modules():
    if isinstance(module, Attention):
        module.set_kv_cache(kv_cache)
```

这样更符合真实 runtime 初始化依赖。

---

# 21. Step 20：PagedKVCache 加 `block_size` 属性

02-A 的：

```python
class PagedKVCache(nn.Module):
```

建议补：

```python
self.num_layers = num_layers
self.num_blocks = num_blocks
self.block_size = block_size
self.num_kv_heads = num_kv_heads
self.head_dim = head_dim
```

方便 model/runner 做 invariant check。

---

# 22. Step 21：Tokenizer 接入 LLMEngine

现在才把公共 API 从：

```python
generate_token_ids([[...], [...]])
```

扩展到字符串。

在 `LLMEngine.__init__()`：

```python
from transformers import AutoTokenizer

self.tokenizer = AutoTokenizer.from_pretrained(
    self.config.model,
    trust_remote_code=True,
)
self.config.eos_token_id = self.tokenizer.eos_token_id
```

增加：

```python
def generate(
    self,
    prompts: list[str],
    sampling_params,
):
    prompt_token_ids = [
        self.tokenizer.encode(p, add_special_tokens=False)
        for p in prompts
    ]

    outputs = self.generate_token_ids(
        prompt_token_ids,
        sampling_params,
    )

    for out in outputs:
        out["text"] = self.tokenizer.decode(
            out["token_ids"],
            skip_special_tokens=True,
        )

    return outputs
```

是否套 chat template，应当由更高层显式决定，不要在底层偷偷加。

---

# 23. Step 22：第一个真实模型测试不是 generate，而是“单层 shape test”

新建：

```text
tests/test_qwen3_shapes.py
```

先用极小 fake config：

```text
hidden_size=32
num_heads=4
num_kv_heads=2
head_dim=8
intermediate_size=64
num_hidden_layers=2
vocab_size=128
```

建立一个 CPU PagedKVCache：

```text
num_blocks=8
block_size=4
```

构造一个 mixed batch metadata，跑：

```python
hidden = model(input_ids, positions)
assert hidden.shape == (num_flat_tokens, hidden_size)
```

这一步不加载 checkpoint。

它只验证：

```text
Q/K/V reshape
GQA repeat
RoPE
residual
MLP
mixed attention
```

的 shape contract。

---

# 24. Step 23：第二个真实模型测试是 weight loader accounting

```python
@pytest.mark.model
def test_loader_loads_every_required_parameter(local_qwen3_path):
    ...
    report = load_weights(...)
    assert report.missing == []
```

并输出：

```text
loaded 311 parameters
packed qkv shards: ...
packed gate_up shards: ...
skipped tied weights: ...
```

loader 应该是“有报告的”，而不是 silent function。

---

# 25. Step 24：第三个测试才是 Hugging Face logits equivalence

这是 Day05 最关键的 correctness 验收。

先暂时关闭：

```text
Prefix Cache
mixed batching
随机 sampling
```

构造单请求：

```text
prompt = 一小段文本
```

HF：

```python
with torch.no_grad():
    hf_logits = hf_model(input_ids).logits[:, -1]
```

tinyInfer：

```text
整段 prompt 做一次 cold prefill
取最后 query hidden
LM head
```

比较：

```python
torch.testing.assert_close(
    tiny_logits.float(),
    hf_logits.float(),
    atol=..., rtol=...,
)
```

BF16 下 tolerance 不应设成 `1e-8`。

先观察 max absolute / relative error，再定合理阈值。

---

# 26. Step 25：再做 greedy token-by-token equivalence

如果 logits 对齐，再跑：

```text
prompt → greedy 8 tokens
```

比较：

```python
assert tiny_tokens == hf_tokens
```

如果 logits 有小数值误差但 argmax 一直一致，这通常可以接受；但你仍然应该记录误差。

---

# 27. Step 26：然后恢复 Prefix Cache，做“输出不变”测试

准备两个 prompt：

```text
A = shared_system_prompt + question_A
B = shared_system_prompt + question_B
```

运行：

```text
Case 1: 禁用 cache 或 cold run B
Case 2: 先 A，再 B，B 命中 prefix
```

Greedy 下：

```python
assert output_B_cached == output_B_cold
```

同时：

```python
assert seq_B.num_cached_tokens > 0
```

这是 persistent prefix cache 的端到端 correctness test。

---

# 28. Step 27：最后才恢复 mixed batching，做输出一致性测试

准备多个 prompt。

Case A：

```text
max_num_seqs=1
```

逐条跑，得到 baseline token ids。

Case B：

```text
max_num_seqs>1
small max_num_batched_tokens
→ 强迫 chunked prefill + mixed decode
```

Greedy 下：

```python
assert batched_outputs == baseline_outputs
```

注意比较顺序用原 request id，而不是完成顺序。

如果这一步失败，不要立刻怀疑模型权重；最可能出错的是：

```text
slot_mapping
context_len
causal offset
block_table
postprocess progress
```

这就是为什么 02-A 要先有 metadata 单测。

---

# 29. Day05 完成标准

你至少应该通过四层 correctness：

```text
Layer 1: shape
Layer 2: loader accounting
Layer 3: single-request logits / greedy 与 HF 对齐
Layer 4: Prefix Cache + mixed batch 不改变 greedy 输出
```

只有全部通过，才进入 Day06。

---

# Day 06：工程闭环——测试、观测、Benchmark、Profile 与后续优化基线

# 30. Day06 的目标

Day06 不继续疯狂加 feature。

要把项目从：

```text
“我好像跑起来了”
```

升级成：

```text
“我能证明它基本正确，并能量化下一步该优化哪里”
```

今天做：

```text
1. final invariant system
2. structured trace
3. benchmark contract
4. TTFT / ITL / E2E
5. prefix-cache metrics
6. mixed-batch utilization metrics
7. HF correctness regression
8. profiling hooks
```

---

# 31. Step 1：整理最终 Day01–06 目录树

推荐：

```text
tinyInfer/
├── README.md
├── pyproject.toml
├── benchmarks/
│   ├── benchmark_latency.py
│   ├── benchmark_throughput.py
│   └── workloads.py
├── docs/
│   ├── 01-tinyInfer-control-plane-day01-02.md
│   ├── 02-tinyInfer-kv-modelrunner-day03-04.md
│   ├── 02-A-tinyInfer-runtime-consolidation.md
│   └── 03-tinyInfer-qwen3-runtime-day05-06-revised.md
├── examples/
│   ├── basic_generate.py
│   └── mixed_batch_trace.py
├── tests/
│   ├── conftest.py
│   ├── test_sampling_params.py
│   ├── test_sequence.py
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   ├── test_model_runner_meta.py
│   ├── test_attention.py
│   ├── test_linear.py
│   ├── test_rope.py
│   ├── test_qwen3_shapes.py
│   ├── test_loader.py
│   ├── test_engine.py
│   └── test_hf_equivalence.py
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
    │   ├── activation.py
    │   ├── attention.py
    │   ├── embed_head.py
    │   ├── layernorm.py
    │   ├── linear.py
    │   ├── rotary_embedding.py
    │   └── sampler.py
    ├── models/
    │   ├── __init__.py
    │   └── qwen3.py
    └── utils/
        ├── __init__.py
        ├── context.py
        ├── debug.py
        └── loader.py
```

---

# 32. Step 2：给 Sequence 加 timestamps

修改：

```text
tinyinfer/engine/sequence.py
```

加入：

```python
import time
```

在 `__init__()`：

```python
self.arrival_time = time.perf_counter()
self.first_scheduled_time = None
self.first_token_time = None
self.finished_time = None
self.output_token_times = []
```

首次真正被 schedule：

```python
if seq.first_scheduled_time is None:
    seq.first_scheduled_time = time.perf_counter()
```

首次 sample：

```python
now = time.perf_counter()
if seq.first_token_time is None:
    seq.first_token_time = now
seq.output_token_times.append(now)
```

finish：

```python
seq.finished_time = time.perf_counter()
```

---

# 33. Step 3：明确定义 TTFT / ITL / E2E

```text
TTFT = first_token_time - arrival_time

E2E = finished_time - arrival_time

ITL_i = output_token_times[i] - output_token_times[i-1]
```

不要把：

```text
prefill kernel time
```

叫 TTFT。

TTFT 包含：

```text
queueing
scheduler wait
prefill
sampling
```

如果想分开，再额外记：

```text
time_in_queue
prefill_compute_time
```

---

# 34. Step 4：Benchmark contract

新建：

```text
benchmarks/workloads.py
```

先定义 deterministic workload：

```python
@dataclass(slots=True)
class RequestSpec:
    prompt: str
    max_tokens: int
```

推荐至少三类：

```text
short:
    prompt 32~64 tokens
    output 32

long_prefill:
    prompt 2K~4K
    output 32

shared_prefix:
    大 system prompt + 不同 question
```

不要一开始用随机 prompt 长度，否则每次跑结果都不好对比。

---

# 35. Step 5：吞吐 benchmark

新建：

```text
benchmarks/benchmark_throughput.py
```

记录：

```text
wall_time
prompt_tokens_total
output_tokens_total
requests_total
```

计算：

```text
output tokens/s
all processed tokens/s
requests/s
```

为什么至少报两个 token throughput？

因为 Prefix Cache 会改变实际 compute 的 prompt token 数；只看“输入 token 总数 / s”可能会产生误导。

额外记录：

```text
actual_computed_prompt_tokens
cached_prompt_tokens
```

---

# 36. Step 6：Latency benchmark

新建：

```text
benchmarks/benchmark_latency.py
```

至少输出：

```text
TTFT p50/p90/p99
ITL p50/p90/p99
E2E p50/p90/p99
```

教学版如果不想依赖 numpy，可以：

```python
statistics.median
```

percentile 自己写小 helper。

---

# 37. Step 7：加入 Prefix Cache 指标

从 `BlockManager.stats()` 读取：

```text
cache_hits
cache_misses
evictions
cached_blocks
active_blocks
free_blocks
```

进一步增加 token-level：

```text
prefix_cached_tokens_total
prompt_tokens_total
```

得到：

```text
prefix cache token hit rate
= cached_prompt_tokens / prompt_tokens_total
```

但注意：为了取得 logits 而重算的最后一个 block/token，在统计时要明确你的定义。

---

# 38. Step 8：加入 mixed batching utilization 指标

Scheduler 每轮记录：

```text
scheduled_tokens
num_decode_tokens
num_prefill_tokens
num_items
```

然后：

```text
budget utilization
= scheduled_tokens / max_num_batched_tokens
```

另外统计：

```text
mixed_steps
prefill_only_steps
decode_only_steps
```

这样你才能量化：

```text
Mixed Batching 到底有没有真正发生？
```

而不是只看代码里“支持”。

---

# 39. Step 9：Runtime invariant checker 升级

在 debug 模式下每轮检查：

## Sequence invariants

```text
0 <= num_cached_tokens <= num_prompt_tokens
0 <= num_computed_tokens <= num_tokens
0 <= num_scheduled_tokens
```

waiting prefill：

```text
num_computed_tokens <= num_prompt_tokens
```

running decode：

```text
num_computed_tokens >= num_prompt_tokens
num_tokens - num_computed_tokens ∈ {0,1}
```

正常 autoregressive loop 中通常是 1。

## Block invariants

```text
active block 不在 free_lru
ref=0 block 必须在 free_lru
cache index 中的 block.hash 必须匹配 key
block table id 必须合法
```

## SchedulerOutput invariants

```text
sum(num_tokens) <= max_num_batched_tokens
len(items) <= max_num_seqs
item.start_pos == seq.num_computed_tokens at schedule snapshot
item.start_pos + item.num_tokens <= seq.num_tokens
```

---

# 40. Step 10：Structured trace，不再散落 print

推荐最小事件结构：

```python
@dataclass(slots=True)
class TraceEvent:
    ts: float
    name: str
    seq_id: int | None
    fields: dict
```

先支持：

```text
request_add
schedule
prefix_hit
block_alloc
block_evict
forward_begin
forward_end
sample
request_finish
```

输出 JSONL：

```text
trace.jsonl
```

以后你可以直接用 Python/pandas 画 timeline，而不需要解析人类日志。

---

# 41. Step 11：给 loader 保留独立 report

建议：

```python
@dataclass(slots=True)
class LoadReport:
    loaded: list[str]
    missing: list[str]
    unexpected: list[str]
    packed: list[str]
```

初始化结束直接：

```python
assert not report.missing
```

这样后面更换 Qwen3 版本时，如果参数名发生变化，会立刻暴露。

---

# 42. Step 12：正确的 benchmark matrix

至少跑下面四组：

```text
A. single request, no shared prefix
B. many independent requests
C. shared long prefix
D. long prefill + many active decodes
```

对 C：

```text
cold cache
warm cache
```

对 D：

```text
较大 token budget
较小 token budget
```

重点观察：

```text
throughput
TTFT
ITL
cache hit rate
mixed-step ratio
budget utilization
```

---

# 43. Step 13：为什么 Day06 暂时不加入 CUDA Graph？

原 03 把 CUDA Graph 放在 Day05。

新版建议后移。

原因不是 CUDA Graph 不重要，而是：

```text
你刚刚把 batch 从 homogeneous 改成 mixed
q_lens / context_lens / block tables 都是动态 metadata
```

此时先建立 eager baseline 更重要。

CUDA Graph 需要额外解决：

```text
固定 tensor buffer
固定或 bucketed batch shape
copy metadata into static buffer
replay output ownership
```

如果现在一起加，一旦结果不一致你很难判断：

```text
模型错
metadata 错
还是 graph buffer 错
```

因此建议将 CUDA Graph 放到未来独立章节，先有可对照的 eager baseline。

---

# 44. Step 14：为什么 Day06 也暂时不做 Tensor Parallel？

同理：

```text
单卡 Qwen3 eager correctness
```

是所有后续优化的 reference implementation。

TP 应该要求：

```text
TP=1 tinyInfer
TP=2 tinyInfer
```

在相同 greedy 输入下 token 对齐。

没有 TP=1 reference，就很难定位：

```text
sharding 错了
all-reduce 错了
还是模型本身就错了
```

所以新版 Day05–06 把 TP 从“必须同天完成”降级成后续优化章节。

---

# 45. Step 15：建议加一个 `--debug-correctness` 模式

在 Config 增加：

```python
debug_correctness: bool = False
```

开启时：

```text
1. enable invariant checks
2. disable CUDA Graph
3. use eager attention path
4. optional float32 reference checks
5. strict loader
6. deterministic greedy sampling
```

这样你后续加 Triton/FlashAttention/Graph 时，可以随时退回 reference path。

---

# 46. Step 16：和 Hugging Face 对齐时的排错顺序

如果 greedy 第一个 token 就不一致：

```text
1. tokenizer input_ids 是否完全相同？
2. embedding 输出是否相同？
3. 第 0 层 norm 是否相同？
4. q/k/v projection 是否相同？
5. RoPE 后 q/k 是否相同？
6. attention output 是否相同？
7. MLP output 是否相同？
8. final norm 是否相同？
9. lm_head logits 是否相同？
```

不要直接打印整模型几 GB tensor。

每层只记录：

```text
shape
max_abs
mean_abs
sample few elements
```

---

# 47. Step 17：Prefix Cache 出错时的排错顺序

如果 cold run 正确，warm prefix-cache run 错：

优先看：

```text
1. cached block token ids 是否真匹配？
2. cached block physical id 是否仍然保存原 KV？
3. free_lru hit 时是否正确移除？
4. 是否错误 eviction 了仍 active block？
5. start_pos / num_computed_tokens 是否正确？
6. slot_mapping 是否覆盖到了正确 suffix？
7. context_len 是否包含 cached prefix？
8. explicit causal mask 是否按绝对 position 构造？
```

模型权重通常不是第一嫌疑。

---

# 48. Step 18：Mixed batching 出错时的排错顺序

如果 single-request 正确，mixed batch 错：

```text
1. flat input_ids 顺序
2. cu_seqlens_q 边界
3. 每 seq block_table row
4. context_lens
5. slot_mapping
6. sample flat index
7. postprocess sampled token 回填 seq_id
```

尤其注意：

```text
sampled tokens 不能再按“所有 item 一一对应” zip 回去
```

因为 partial prefill item 没有 sampled token。

---

# 49. Step 19：Day06 最终自动化测试命令

建议写在 README：

```bash
# fast CPU/unit suite
python -m compileall -q tinyinfer
python -m pytest -m 'not model'

# local model correctness suite
TINYINFER_MODEL=/path/to/Qwen3-0.6B \
python -m pytest -m model
```

如果愿意，可以在 `conftest.py`：

```python
import os
import pytest


@pytest.fixture
def local_model_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path
```

这样普通单测不会被本地模型环境绑架。

---

# 50. Step 20：最小 README 验收说明

README 至少应该能让未来的你执行：

```bash
python -m pip install --user -e '.[dev]'
python -m pytest -m 'not model'

TINYINFER_MODEL=/path/to/Qwen3-0.6B \
python examples/basic_generate.py
```

并解释：

```text
- 当前只支持单卡 eager
- mixed batching 已实现
- chunked prefill 已实现
- persistent prefix cache 已实现
- PagedAttention 当前是教学/reference 实现，不代表高性能 kernel
```

这比写“高性能推理框架”更客观。

---

# 51. Day05–06 最终白板题

完成新版 03 后，你应该能不看代码回答：

### Q1. 为什么真实模型接入之前一定要先把 `num_computed_tokens` 做对？

因为 ModelRunner 的输入起点、slot mapping、context length、chunked prefill 和 decode 都依赖同一个“KV frontier”。

### Q2. 为什么 mixed batch 不要求 Attention 写两套完全分离的 forward？

因为本质上每条请求都是：

```text
本轮新增 q tokens
对完整历史 KV 做 causal attention
```

区别主要是 `q_len` 与 query start position。

### Q3. 为什么 partial prefill 不应该做 LM head / sample？

因为 prompt 还没完整消费；中间 query 的 logits 不用于 autoregressive generation。

### Q4. Prefix Cache 为什么只存 KV 不能直接给出 next token？

因为 next-token logits 需要最后 query 的 hidden state 经 final norm / LM head；KV cache 本身不保存该输出 hidden state。

### Q5. 为什么 request finish 后 block 可以 ref_count=0 但仍是有效 cache？

refcount 表示当前 ownership，不表示物理内容是否失效。

### Q6. 为什么 eviction 应延迟到 physical block 被真正复用？

因为否则 cache 无法跨请求生命周期存在。

### Q7. 为什么 HF logits equivalence 要早于 throughput benchmark？

性能数据只有建立在正确结果上才有意义。

### Q8. 为什么新版 Day05–06 暂时后移 CUDA Graph / TP？

先建立单卡 eager reference，使后续每个性能优化都有 correctness oracle。

---

# 52. Day06 最终 commit 建议

```bash
git add .
git commit -m "feat: add qwen3 model layers and eager runtime"

git add .
git commit -m "feat: add strict safetensors loader"

git add .
git commit -m "feat: run real qwen3 on mixed paged runtime"

git add .
git commit -m "test: align qwen3 logits and greedy outputs with transformers"

git add .
git commit -m "test: verify prefix cache and mixed batching output equivalence"

git add .
git commit -m "bench: add latency throughput and cache metrics"

git add .
git commit -m "debug: add runtime invariants and structured tracing"
```

---

# 53. 新版 Day01–06 完成后，你真正拥有的是什么？

不是“写了一个缩小版 vLLM”这么简单，而是已经亲手建立了下面这条完整因果链：

```text
request
  ↓
Sequence lifecycle
  ↓
continuous / chunked / mixed scheduling
  ↓
logical token positions
  ↓
physical KV blocks
  ↓
persistent prefix cache + eviction
  ↓
flat mixed-batch metadata
  ↓
Qwen3 projection + RoPE
  ↓
paged causal attention
  ↓
selective logits
  ↓
sampling
  ↓
postprocess / next decode token
```

而且每一层都有可以独立执行的 correctness test。

这才是后续继续做：

```text
FlashAttention / Triton
CUDA Graph
Tensor Parallel
更复杂 scheduler
Speculative Decoding
```

时最有价值的基础。
