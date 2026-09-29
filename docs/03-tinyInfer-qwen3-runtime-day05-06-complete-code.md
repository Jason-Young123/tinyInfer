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

<!-- 03-CODE-COMPLETE: step1-config -->

### 本步骤完整代码：`tinyinfer/config.py`

```python
from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # Hugging Face repo id 或本地模型目录。
    model: str | None = None

    # Scheduler/runtime 配置。
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 64
    max_model_len: int = 4096

    # 显存与执行模式配置。
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = True

    # Paged KV Cache 配置。
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = 0

    # 真实模型接入后由 load_hf_config()/tokenizer 填充。
    hf_config: Any | None = None
    eos_token_id: int = -1

    # Day06 correctness/debug 开关；此时先保留字段，后面会正式使用。
    debug_correctness: bool = False

    def validate_runtime_fields(self) -> None:
        """检查与模型无关的 runtime 配置。"""
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.kvcache_block_size <= 0:
            raise ValueError("kvcache_block_size must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")

    def load_hf_config(self) -> None:
        """读取真实 Hugging Face 模型配置，并同步 tinyInfer 的长度上限。"""
        if self.model is None:
            raise ValueError("model path/name is required for real runtime")

        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=True,
        )

        hf_max_len = getattr(self.hf_config, "max_position_embeddings", None)
        if hf_max_len is not None:
            self.max_model_len = min(self.max_model_len, int(hf_max_len))
```

### 本步骤完整测试：`tests/test_config.py`

```python
import os

import pytest

from tinyinfer.config import Config


def test_config_runtime_validation():
    cfg = Config(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=1024,
        kvcache_block_size=16,
    )
    cfg.validate_runtime_fields()


@pytest.mark.model
def test_config_model():
    # 使用环境变量而不是把个人本地路径硬编码进仓库。
    model_path = os.getenv("TINYINFER_MODEL")
    if not model_path:
        pytest.skip("TINYINFER_MODEL is not set")

    cfg = Config(model=model_path)
    cfg.load_hf_config()

    # Qwen3 runtime 后续会依赖这些字段，因此这里显式检查。
    assert cfg.hf_config.hidden_size > 0
    assert cfg.hf_config.num_hidden_layers > 0
    assert cfg.hf_config.num_attention_heads > 0
    assert cfg.hf_config.num_key_value_heads > 0

    head_dim = getattr(
        cfg.hf_config,
        "head_dim",
        cfg.hf_config.hidden_size // cfg.hf_config.num_attention_heads,
    )
    assert head_dim > 0

    print("hidden_size =", cfg.hf_config.hidden_size)
    print("num_hidden_layers =", cfg.hf_config.num_hidden_layers)
    print("num_attention_heads =", cfg.hf_config.num_attention_heads)
    print("num_key_value_heads =", cfg.hf_config.num_key_value_heads)
    print("head_dim =", head_dim)
```

运行：

```bash
pytest -v -s tests/test_config.py
```
<!-- /03-CODE-COMPLETE: step1-config -->


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

<!-- 03-CODE-COMPLETE: linear -->

### 截至 Step 4，`tinyinfer/layers/linear.py` 完整代码

```python
import torch
import torch.nn.functional as F
from torch import nn


class LinearBase(nn.Module):
    """tinyInfer 单卡 correctness baseline 使用的最普通 Linear。"""

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

        # F.linear(x, weight) 约定 weight 为 [out_features, in_features]。
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
            # 显式注册名为 bias 的空参数，使模块接口与 nn.Linear 一致。
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(LinearBase):
    """把 Q/K/V 三个 projection 打包成一次大 Linear。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
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

    def split_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 [..., q_size + kv_size + kv_size] 切回 Q/K/V。"""
        q, k, v = torch.split(
            x,
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        return q, k, v


class MergedGateUpLinear(LinearBase):
    """把 SwiGLU 的 gate_proj/up_proj 打包成一次 Linear。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        **kwargs,
    ):
        self.intermediate_size = intermediate_size
        super().__init__(
            hidden_size,
            2 * intermediate_size,
            **kwargs,
        )

    def split_gate_up(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 最后一维结构固定为 [gate | up]，因此平均切成两份。
        gate, up = x.chunk(2, dim=-1)
        return gate, up
```

### 完整测试：`tests/test_linear.py`

```python
import torch

from tinyinfer.layers.linear import (
    LinearBase,
    MergedGateUpLinear,
    QKVParallelLinear,
)


def test_linear_base_shape():
    layer = LinearBase(10, 24, bias=False)
    x = torch.randn(5, 10)
    y = layer(x)
    assert y.shape == (5, 24)


def test_qkv_parallel_linear_shapes():
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    layer = QKVParallelLinear(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        bias=False,
    )

    x = torch.randn(7, hidden_size)
    packed = layer(x)
    q, k, v = layer.split_qkv(packed)

    assert packed.shape == (7, 64)  # 32 + 16 + 16
    assert q.shape == (7, 32)
    assert k.shape == (7, 16)
    assert v.shape == (7, 16)


def test_merged_gate_up_linear_shapes():
    layer = MergedGateUpLinear(
        hidden_size=10,
        intermediate_size=24,
        bias=False,
    )

    x = torch.randn(5, 10)
    packed = layer(x)
    gate, up = layer.split_gate_up(packed)

    assert packed.shape == (5, 48)
    assert gate.shape == (5, 24)
    assert up.shape == (5, 24)
```
<!-- /03-CODE-COMPLETE: linear -->


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

<!-- 03-CODE-COMPLETE: rmsnorm -->

### `tinyinfer/layers/layernorm.py` 完整代码

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Qwen/Llama 风格 RMSNorm 的 correctness reference 实现。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # reduction 在 FP32 做，先优先保证数值稳定性。
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)

        # 再转换回模型原 dtype，与可训练缩放参数逐元素相乘。
        return self.weight * x_norm.to(input_dtype)
```

### 完整测试：`tests/test_layernorm.py`

```python
import torch

from tinyinfer.layers.layernorm import RMSNorm


def test_rmsnorm_shape_and_finite():
    norm = RMSNorm(16, eps=1e-6)
    x = torch.randn(3, 5, 16)
    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rmsnorm_matches_manual_reference():
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.fill_(1.0)

    x = torch.randn(4, 8)
    y = norm(x)

    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref.to(x.dtype)

    torch.testing.assert_close(y, ref)
```
<!-- /03-CODE-COMPLETE: rmsnorm -->


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

<!-- 03-CODE-COMPLETE: activation -->

### `tinyinfer/layers/activation.py` 完整代码

```python
import torch
from torch import nn


class SiluAndMul(nn.Module):
    """SwiGLU：SiLU(gate) * up。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("SiluAndMul expects an even last dimension")

        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
```

### 完整测试：`tests/test_activation.py`

```python
import torch

from tinyinfer.layers.activation import SiluAndMul


def test_silu_and_mul_shape():
    x = torch.randn(7, 2 * 16)
    y = SiluAndMul()(x)
    assert y.shape == (7, 16)


def test_silu_and_mul_matches_reference():
    x = torch.randn(5, 48)
    gate, up = x.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up

    y = SiluAndMul()(x)
    torch.testing.assert_close(y, ref)
```
<!-- /03-CODE-COMPLETE: activation -->


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

<!-- 03-CODE-COMPLETE: rope -->

### `tinyinfer/layers/rotary_embedding.py` 完整代码

```python
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Qwen/Llama half-split rotary layout 的教学/reference 实现。

    输入：
        q: [num_tokens, num_q_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        positions: [num_tokens]

    注意 mixed batch 中 positions 不要求单调，因为每个 flat token 都带自己的
    absolute sequence position。
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int | None = None,
        base: float = 10000.0,
    ):
        super().__init__()

        if rotary_dim is None:
            rotary_dim = head_dim
        if rotary_dim <= 0:
            raise ValueError("rotary_dim must be positive")
        if rotary_dim > head_dim:
            raise ValueError("rotary_dim cannot exceed head_dim")
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )

        # inv_freq 是模型状态，但不是 checkpoint parameter。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """对 x 的前 rotary_dim 维应用 half-split RoPE。"""
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        half = self.rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]

        # cos/sin: [T, half] -> [T, 1, half]，广播到所有 heads。
        cos = cos.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        sin = sin.unsqueeze(1).to(dtype=x.dtype, device=x.device)

        # 等价于 HF rotate_half 的 half-split layout：
        # [x1, x2] -> [-x2, x1]。
        first = x1 * cos - x2 * sin
        second = x2 * cos + x1 * sin
        rotated = torch.cat((first, second), dim=-1)

        if x_pass.shape[-1] == 0:
            return rotated
        return torch.cat((rotated, x_pass), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("q/k must be [num_tokens, num_heads, head_dim]")
        if positions.ndim != 1:
            raise ValueError("positions must be a 1-D tensor")
        if q.shape[0] != k.shape[0] or q.shape[0] != positions.numel():
            raise ValueError("q, k and positions must have the same token count")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("q/k head_dim mismatch")

        positions_fp32 = positions.to(device=q.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=q.device)

        # [T] outer [rotary_dim/2] -> [T, rotary_dim/2]
        freqs = torch.outer(positions_fp32, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k
```

### `tests/test_rope.py` 完整代码

```python
import torch

from tinyinfer.layers.rotary_embedding import RotaryEmbedding


def reference_apply_rotary(x, positions, base=10000.0):
    """独立写一个小 reference，避免测试直接复用被测函数内部实现。"""
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    freqs = torch.outer(positions.float(), inv_freq)
    cos = freqs.cos().unsqueeze(1).to(x.dtype)
    sin = freqs.sin().unsqueeze(1).to(x.dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat(
        (x1 * cos - x2 * sin, x2 * cos + x1 * sin),
        dim=-1,
    )


def test_rope_mixed_positions_match_reference():
    torch.manual_seed(0)

    rope = RotaryEmbedding(head_dim=8, rotary_dim=8, base=10000.0)
    q = torch.randn(6, 4, 8)
    k = torch.randn(6, 2, 8)

    # decode/prefill/decode 混排，positions 故意不是单调递增。
    positions = torch.tensor([12, 4, 5, 6, 7, 29], dtype=torch.long)

    q_out, k_out = rope(q, k, positions)
    q_ref = reference_apply_rotary(q, positions)
    k_ref = reference_apply_rotary(k, positions)

    torch.testing.assert_close(q_out, q_ref)
    torch.testing.assert_close(k_out, k_ref)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(3, 4, 8)
    k = torch.randn(3, 2, 8)
    positions = torch.zeros(3, dtype=torch.long)

    q_out, k_out = rope(q, k, positions)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_rope_shape_is_preserved():
    rope = RotaryEmbedding(head_dim=8)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    positions = torch.arange(5)

    q_out, k_out = rope(q, k, positions)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
```

运行：

```bash
pytest -v tests/test_rope.py
```
<!-- /03-CODE-COMPLETE: rope -->


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

<!-- 03-CODE-COMPLETE: embed-head -->

### `tinyinfer/layers/embed_head.py` 完整代码

```python
import torch
from torch import nn


class VocabEmbedding(nn.Embedding):
    """单卡 baseline 直接使用 PyTorch Embedding。"""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids)


class LMHead(nn.Linear):
    """把 hidden state 投影到 vocabulary logits。"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            vocab_size,
            bias=bias,
            **kwargs,
        )
```

### 完整测试：`tests/test_embed_head.py`

```python
import torch

from tinyinfer.layers.embed_head import LMHead, VocabEmbedding


def test_embedding_and_lm_head_shapes():
    vocab_size = 128
    hidden_size = 32

    embedding = VocabEmbedding(vocab_size, hidden_size)
    lm_head = LMHead(hidden_size, vocab_size, bias=False)

    input_ids = torch.tensor([1, 5, 9, 11], dtype=torch.long)
    hidden = embedding(input_ids)
    logits = lm_head(hidden)

    assert hidden.shape == (4, hidden_size)
    assert logits.shape == (4, vocab_size)
```
<!-- /03-CODE-COMPLETE: embed-head -->


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

<!-- 03-CODE-COMPLETE: sampler -->

### `tinyinfer/sampling_params.py` 完整代码

```python
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    greedy: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # greedy=True 时 temperature 不参与采样；仍要求非负以避免脏配置。
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not self.greedy and self.temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0 when greedy=False"
            )
```

### `tinyinfer/layers/sampler.py` 完整代码

```python
import torch
from torch import nn


class Sampler(nn.Module):
    """支持同一个 mixed batch 中每条请求独立 greedy/temperature。"""

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must be [num_samples, vocab_size]")
        if temperatures.shape != (logits.shape[0],):
            raise ValueError("temperatures must be [num_samples]")

        n = logits.shape[0]
        if greedy_mask is None:
            greedy_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
        if greedy_mask.shape != (n,):
            raise ValueError("greedy_mask must be [num_samples]")

        result = torch.empty(n, dtype=torch.long, device=logits.device)

        # Greedy 行直接 argmax。
        if greedy_mask.any():
            result[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

        # 非 greedy 行再做 temperature sampling。
        sample_mask = ~greedy_mask
        if sample_mask.any():
            temps = temperatures[sample_mask]
            if torch.any(temps <= 0):
                raise ValueError("non-greedy temperatures must be > 0")

            scaled = logits[sample_mask] / temps.unsqueeze(-1)
            probs = torch.softmax(scaled.float(), dim=-1)
            result[sample_mask] = torch.multinomial(
                probs,
                num_samples=1,
            ).squeeze(-1)

        return result
```

### 完整测试：`tests/test_sampler.py`

```python
import torch

from tinyinfer.layers.sampler import Sampler


def test_sampler_greedy():
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 2.0, 0.3],
            [4.0, 1.0, 3.0],
        ]
    )
    temperatures = torch.ones(2)
    greedy = torch.tensor([True, True])

    out = sampler(logits, temperatures, greedy)
    assert out.tolist() == [1, 0]


def test_sampler_mixed_greedy_and_sampling_shapes():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.randn(4, 32)
    temperatures = torch.tensor([1.0, 0.8, 1.0, 0.7])
    greedy = torch.tensor([True, False, True, False])

    out = sampler(logits, temperatures, greedy)
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert 0 <= int(out.min())
    assert int(out.max()) < 32
```
<!-- /03-CODE-COMPLETE: sampler -->


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

<!-- 03-CODE-COMPLETE: qwen3 -->

### 截至 Step 13，`tinyinfer/models/qwen3.py` 完整代码

```python
import torch
from torch import nn

from tinyinfer.layers.activation import SiluAndMul
from tinyinfer.layers.attention import Attention
from tinyinfer.layers.embed_head import LMHead, VocabEmbedding
from tinyinfer.layers.layernorm import RMSNorm
from tinyinfer.layers.linear import (
    LinearBase,
    MergedGateUpLinear,
    QKVParallelLinear,
)
from tinyinfer.layers.rotary_embedding import RotaryEmbedding


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int, kv_cache=None):
        super().__init__()

        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        )
        self.q_size = self.num_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=bool(getattr(config, "attention_bias", False)),
        )

        # Qwen3 使用 per-head Q/K RMSNorm；这是与 Qwen2 不能混淆的地方。
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            rotary_dim=self.head_dim,
            base=float(getattr(config, "rope_theta", 10000.0)),
        )

        self.attn = Attention(
            layer_idx=layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim ** -0.5,
            kv_cache=kv_cache,
            block_size=int(getattr(config, "kvcache_block_size", 256)),
        )

        self.o_proj = LinearBase(
            self.q_size,
            config.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
        )

    def set_kv_cache(self, kv_cache) -> None:
        self.attn.set_kv_cache(kv_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [num_flat_tokens, hidden_size]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Qwen3 的 q_norm/k_norm 只沿每个 head 的 head_dim 做 RMSNorm。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # positions 直接来自 mixed-batch metadata。
        q, k = self.rotary_emb(q, k, positions)

        # Attention 统一处理 prefill/decode/mixed；输出 [T, Hq, D]。
        out = self.attn(q, k, v)
        out = out.reshape(-1, self.q_size)
        return self.o_proj(out)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.act(x)
        return self.down_proj(x)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, kv_cache=None):
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

    def set_kv_cache(self, kv_cache) -> None:
        self.self_attn.set_kv_cache(kv_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config, kv_cache=None):
        super().__init__()
        self.embed_tokens = VocabEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, i, kv_cache)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def set_kv_cache(self, kv_cache) -> None:
        for layer in self.layers:
            layer.set_kv_cache(kv_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config, kv_cache=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, kv_cache)
        self.lm_head = LMHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # 若 checkpoint 使用 tied embeddings，让两个 module 共享同一个 Parameter。
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_kv_cache(self, kv_cache) -> None:
        self.model.set_kv_cache(kv_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 只返回 hidden states；mixed batch 中并非所有 token 都需要 logits。
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
```

> 注：上面已经显式加入 Qwen3 的 `q_norm/k_norm`。如果你本地 checkpoint 的 Hugging Face 实现或 `config.json` 还包含额外 RoPE scaling 变体，应以本地版本为准，再扩展 `RotaryEmbedding`；不要静默忽略配置字段。
<!-- /03-CODE-COMPLETE: qwen3 -->


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

<!-- 03-CODE-COMPLETE: model-runner-step14 -->

### 截至 Step 14，`tinyinfer/engine/model_runner.py` 完整代码

```python
import torch

from tinyinfer.engine.scheduler import SchedulerOutput
from tinyinfer.layers.sampler import Sampler
from tinyinfer.utils.context import reset_context, set_context


def token_to_slot(seq, token_position: int) -> int:
    """把 sequence 内的逻辑 token position 映射到 paged KV 的 flat slot。"""
    block_size = seq.block_size
    logical_block = token_position // block_size
    offset = token_position % block_size

    if logical_block >= len(seq.block_table):
        raise RuntimeError("token position has no allocated physical block")

    physical_block = seq.block_table[logical_block]
    return physical_block * block_size + offset


class ToyModelRunner:
    """02-A 控制面测试继续保留的 deterministic runner。"""

    def run(self, output: SchedulerOutput) -> dict[int, int]:
        result = {}
        for item in output.items:
            if item.sample_after:
                result[item.seq.seq_id] = (item.seq.last_token + 1) % 1000
        return result


class ModelRunner:
    """Day05 初版真实 runner；模型对象先从外部注入。"""

    def __init__(self, config, model, device="cuda"):
        self.config = config
        self.model = model
        self.device = torch.device(device)
        self.sampler = Sampler()

    def prepare_batch(
        self,
        output: SchedulerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not output.items:
            raise ValueError("cannot prepare an empty SchedulerOutput")

        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []

        q_lens: list[int] = []
        context_lens: list[int] = []
        is_prefill: list[bool] = []
        sample_mask: list[bool] = []
        block_tables: list[list[int]] = []

        max_blocks = max(len(item.seq.block_table) for item in output.items)

        for item in output.items:
            seq = item.seq
            start = item.start_pos
            end = start + item.num_tokens

            tokens = seq.token_ids[start:end]
            pos = list(range(start, end))

            # Python slice 越界会静默截断，因此必须显式检查。
            if len(tokens) != item.num_tokens:
                raise RuntimeError(
                    "scheduler/model-runner token range mismatch"
                )

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

        input_ids_t = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )
        positions_t = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.device,
        )

        set_context(
            q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(
                context_lens,
                dtype=torch.int32,
                device=self.device,
            ),
            cu_seqlens_q=torch.tensor(
                cu_q,
                dtype=torch.int32,
                device=self.device,
            ),
            max_seqlen_q=max(q_lens, default=0),
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.long,
                device=self.device,
            ),
            block_tables=torch.tensor(
                block_tables,
                dtype=torch.int32,
                device=self.device,
            ),
            is_prefill=torch.tensor(
                is_prefill,
                dtype=torch.bool,
                device=self.device,
            ),
            sample_mask=torch.tensor(
                sample_mask,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        return input_ids_t, positions_t

    @torch.no_grad()
    def run(self, output: SchedulerOutput) -> dict[int, int]:
        try:
            input_ids, positions = self.prepare_batch(output)
            hidden_states = self.model(input_ids, positions)

            # flat hidden_states 中只抽取需要 sample 的每条 sequence 最后一个 query。
            sample_flat_indices: list[int] = []
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

            temperatures = torch.tensor(
                [item.seq.sampling_params.temperature for item in sample_items],
                dtype=logits.dtype,
                device=logits.device,
            )
            greedy_mask = torch.tensor(
                [item.seq.sampling_params.greedy for item in sample_items],
                dtype=torch.bool,
                device=logits.device,
            )

            tokens = self.sampler(
                logits,
                temperatures,
                greedy_mask,
            )

            return {
                item.seq.seq_id: int(token)
                for item, token in zip(sample_items, tokens.tolist())
            }
        finally:
            # Context 是一次 forward 的动态状态，绝不能泄漏到下一轮。
            reset_context()
```
<!-- /03-CODE-COMPLETE: model-runner-step14 -->


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

<!-- 03-CODE-COMPLETE: loader -->

### `tinyinfer/utils/loader.py` 完整代码

```python
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(slots=True)
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    packed: list[str] = field(default_factory=list)
    skipped_tied: list[str] = field(default_factory=list)


def iter_safetensor_weights(model_path: str | Path):
    """按文件名顺序遍历一个本地 HF safetensors checkpoint。"""
    model_path = Path(model_path).expanduser().resolve()
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no .safetensors files found under {model_path}"
        )

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def _copy_checked(param: torch.nn.Parameter, tensor: torch.Tensor, name: str):
    if tuple(param.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"shape mismatch for {name}: "
            f"parameter={tuple(param.shape)} checkpoint={tuple(tensor.shape)}"
        )
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def load_weights(model, model_path: str | Path, strict: bool = True) -> LoadReport:
    """把 HF Qwen3 checkpoint 映射到 tinyInfer packed 参数布局。"""
    params = dict(model.named_parameters())
    loaded_params: set[str] = set()
    report = LoadReport()

    # packed 参数的 shard 名 -> (tinyInfer 子模块名, shard kind)
    packed_attn_suffix = {
        "q_proj.weight": "q",
        "k_proj.weight": "k",
        "v_proj.weight": "v",
    }
    packed_mlp_suffix = {
        "gate_proj.weight": "gate",
        "up_proj.weight": "up",
    }

    for ckpt_name, tensor in iter_safetensor_weights(model_path):
        # ------------------------------------------------------------
        # 1. packed QKV
        # ------------------------------------------------------------
        matched = False
        for suffix, shard in packed_attn_suffix.items():
            if ckpt_name.endswith("self_attn." + suffix):
                prefix = ckpt_name[: -len(suffix)]
                target = prefix + "qkv_proj.weight"
                if target not in params:
                    report.unexpected.append(ckpt_name)
                    matched = True
                    break

                param = params[target]
                module_path = target.rsplit(".weight", 1)[0]
                module = model.get_submodule(module_path)
                q_size = module.q_size
                kv_size = module.kv_size

                if shard == "q":
                    begin, end = 0, q_size
                elif shard == "k":
                    begin, end = q_size, q_size + kv_size
                else:
                    begin, end = q_size + kv_size, q_size + 2 * kv_size

                view = param.data[begin:end]
                if tuple(view.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"packed QKV shape mismatch for {ckpt_name}: "
                        f"target slice={tuple(view.shape)} checkpoint={tuple(tensor.shape)}"
                    )
                view.copy_(tensor.to(device=param.device, dtype=param.dtype))

                # 一个 packed 参数只有 q/k/v 三片都读完才算完整；
                # 这里用 packed 记录 shard，最后单独根据 shard 计数确认。
                report.packed.append(f"{target}:{shard}")
                matched = True
                break
        if matched:
            continue

        # ------------------------------------------------------------
        # 2. packed Gate/Up
        # ------------------------------------------------------------
        for suffix, shard in packed_mlp_suffix.items():
            if ckpt_name.endswith("mlp." + suffix):
                prefix = ckpt_name[: -len(suffix)]
                target = prefix + "gate_up_proj.weight"
                if target not in params:
                    report.unexpected.append(ckpt_name)
                    matched = True
                    break

                param = params[target]
                module_path = target.rsplit(".weight", 1)[0]
                module = model.get_submodule(module_path)
                size = module.intermediate_size

                begin, end = (0, size) if shard == "gate" else (size, 2 * size)
                view = param.data[begin:end]
                if tuple(view.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"packed Gate/Up shape mismatch for {ckpt_name}: "
                        f"target slice={tuple(view.shape)} checkpoint={tuple(tensor.shape)}"
                    )
                view.copy_(tensor.to(device=param.device, dtype=param.dtype))
                report.packed.append(f"{target}:{shard}")
                matched = True
                break
        if matched:
            continue

        # ------------------------------------------------------------
        # 3. tied lm_head：有些 checkpoint 不单独保存它。
        # ------------------------------------------------------------
        if ckpt_name == "lm_head.weight" and ckpt_name not in params:
            report.skipped_tied.append(ckpt_name)
            continue

        # ------------------------------------------------------------
        # 4. 普通 1:1 参数。
        # ------------------------------------------------------------
        if ckpt_name in params:
            _copy_checked(params[ckpt_name], tensor, ckpt_name)
            loaded_params.add(ckpt_name)
            report.loaded.append(ckpt_name)
        else:
            report.unexpected.append(ckpt_name)

    # ------------------------------------------------------------
    # 5. packed 参数完整性 accounting。
    # ------------------------------------------------------------
    packed_shards: dict[str, set[str]] = {}
    for entry in report.packed:
        target, shard = entry.rsplit(":", 1)
        packed_shards.setdefault(target, set()).add(shard)

    for target, shards in packed_shards.items():
        required = {"q", "k", "v"} if target.endswith("qkv_proj.weight") else {"gate", "up"}
        if shards == required:
            loaded_params.add(target)
            report.loaded.append(target)
        elif strict:
            raise RuntimeError(
                f"incomplete packed parameter {target}: got {sorted(shards)}, "
                f"expected {sorted(required)}"
            )

    # tied embeddings 时 named_parameters() 通常只暴露一份 Parameter；
    # 若仍有 lm_head.weight 且它与 embedding 共用 storage，则视为已加载。
    if "lm_head.weight" in params and "model.embed_tokens.weight" in loaded_params:
        if params["lm_head.weight"] is params.get("model.embed_tokens.weight"):
            loaded_params.add("lm_head.weight")
            report.skipped_tied.append("lm_head.weight")

    report.missing = sorted(set(params) - loaded_params)

    if strict:
        if report.unexpected:
            raise RuntimeError(
                "unexpected checkpoint weights: "
                f"{report.unexpected[:20]}"
            )
        if report.missing:
            raise RuntimeError(
                "unloaded model parameters: "
                f"{report.missing[:20]}"
            )

    return report
```
<!-- /03-CODE-COMPLETE: loader -->


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

<!-- 03-CODE-COMPLETE: kv-capacity -->

### 完整容量计算 helper

建议放入 `tinyinfer/engine/model_runner.py`，先作为独立函数：

```python
import torch


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


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


def estimate_num_kv_blocks(
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """模型已经驻留 GPU 后，再根据当前 reserved memory 估算 KV block 数。"""
    if device.type != "cuda":
        # CPU 单测不做显存 profile；要求测试显式给 num_kvcache_blocks。
        if config.num_kvcache_blocks <= 0:
            raise ValueError(
                "CPU runtime requires config.num_kvcache_blocks > 0"
            )
        return config.num_kvcache_blocks

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * config.gpu_memory_utilization)
    reserved = torch.cuda.memory_reserved(device)
    available_for_kv = max(0, budget - reserved)

    per_block = bytes_per_kv_block(
        config.hf_config,
        config.kvcache_block_size,
        dtype,
    )
    if per_block <= 0:
        raise RuntimeError("invalid KV bytes per block")

    num_blocks = available_for_kv // per_block
    if num_blocks <= 0:
        raise RuntimeError(
            "no memory left for KV cache under gpu_memory_utilization"
        )
    return int(num_blocks)
```
<!-- /03-CODE-COMPLETE: kv-capacity -->


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

<!-- 03-CODE-COMPLETE: model-runner-init -->

### Step 19 完整版 `ModelRunner.__init__()` 所需代码

把 Step 14 的 `ModelRunner` 替换为下面版本；`prepare_batch()` / `run()` 保持 Step 14 完整版中的实现不变：

```python
import torch

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights


class ModelRunner:
    def __init__(
        self,
        config,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.sampler = Sampler()

        if self.config.hf_config is None:
            self.config.load_hf_config()

        hf_config = self.config.hf_config

        # tinyInfer 自己的 block_size 不是 HF config 标准字段；
        # 临时挂进去，方便 Qwen3Attention 构造 reference Attention。
        hf_config.kvcache_block_size = self.config.kvcache_block_size

        # 1) 先不绑定 KV cache，构造模型。
        self.model = Qwen3ForCausalLM(
            hf_config,
            kv_cache=None,
        ).to(device=self.device, dtype=self.dtype)

        # 2) 加载真实 checkpoint。
        report = load_weights(
            self.model,
            self.config.model,
            strict=True,
        )
        self.load_report = report
        self.model.eval()

        # 3) 模型参数已经驻留后再估算 KV 容量。
        num_blocks = estimate_num_kv_blocks(
            self.config,
            self.device,
            self.dtype,
        )
        self.config.num_kvcache_blocks = num_blocks

        head_dim = int(
            getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
        )

        # 4) 创建唯一共享的 PagedKVCache。
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.config.kvcache_block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # 5) 后绑定到所有 decoder layers。
        self.model.set_kv_cache(self.kv_cache)
```

为了让这一版成为**真正完整文件**，请把本节上面的 `ModelRunner.__init__()` 与 Step 14 给出的 `token_to_slot()`、`ToyModelRunner`、`prepare_batch()`、`run()` 放在同一个 `tinyinfer/engine/model_runner.py` 中；容量 helper 使用 Step 18 的 `dtype_nbytes()` / `bytes_per_kv_block()` / `estimate_num_kv_blocks()`。
<!-- /03-CODE-COMPLETE: model-runner-init -->


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

<!-- 03-CODE-COMPLETE: attention-final -->

### 截至 Step 20，`tinyinfer/layers/attention.py` 完整代码

```python
import torch
from torch import nn

from tinyinfer.utils.context import get_context


def store_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """把本轮 flat token 的新 K/V 写入 paged physical slots。"""
    if k.shape != v.shape:
        raise ValueError("k/v shape mismatch")
    if k.shape[0] != slot_mapping.numel():
        raise ValueError("slot_mapping length must equal token count")

    for i, slot in enumerate(slot_mapping.tolist()):
        block_id = slot // block_size
        offset = slot % block_size
        cache_k[block_id, offset].copy_(k[i])
        cache_v[block_id, offset].copy_(v[i])


def gather_sequence_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    """按一条 request 的 block_table gather [0, context_len) KV。"""
    if context_len <= 0:
        raise ValueError("context_len must be positive")

    chunks = []
    remaining = context_len

    for block_id in block_table.tolist():
        if block_id < 0 or remaining <= 0:
            break
        take = min(block_size, remaining)
        chunks.append(cache[block_id, :take])
        remaining -= take

    if remaining != 0:
        raise RuntimeError("block_table does not cover context_len")

    return torch.cat(chunks, dim=0)


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

        self.num_layers = int(num_layers)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)

        shape = (
            self.num_layers,
            2,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        # storage 不是模型参数，因此 register_buffer；persistent=False 防止写入 state_dict。
        self.register_buffer(
            "storage",
            torch.empty(shape, dtype=dtype, device=device),
            persistent=False,
        )

    def layer_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("layer_idx out of range")
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]


class Attention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        kv_cache: PagedKVCache | None,
        block_size: int,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.kv_cache = kv_cache
        self.block_size = block_size

    def set_kv_cache(self, kv_cache: PagedKVCache) -> None:
        if kv_cache.block_size != self.block_size:
            raise ValueError("KV cache block_size mismatch")
        self.kv_cache = kv_cache

    def _repeat(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """教学版 materialize GQA repeat，使 Hkv 扩展到 Hq。"""
        if self.num_kv_heads == self.num_heads:
            return k, v

        repeat = self.num_heads // self.num_kv_heads
        return (
            k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1),
        )

    def _attend_one(
        self,
        q_i: torch.Tensor,
        k_hist: torch.Tensor,
        v_hist: torch.Tensor,
        query_start_pos: int,
    ) -> torch.Tensor:
        # q_i: [Tq, Hq, D]
        # k/v_hist: [Tk, Hkv, D]
        q = q_i.transpose(0, 1).unsqueeze(0)
        k = k_hist.transpose(0, 1).unsqueeze(0)
        v = v_hist.transpose(0, 1).unsqueeze(0)

        k, v = self._repeat(k, v)

        tq = q_i.shape[0]
        tk = k_hist.shape[0]

        # 绝对 position causal mask，正确覆盖 cached-prefix + suffix。
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("Attention KV cache has not been bound")

        ctx = get_context()
        required = (
            ctx.cu_seqlens_q,
            ctx.context_lens,
            ctx.slot_mapping,
            ctx.block_tables,
        )
        if any(x is None for x in required):
            raise RuntimeError("incomplete attention Context")

        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx)

        # 必须先写本轮 K/V；这样 gather 的完整 context 已包含本轮 token。
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

            if query_start < 0:
                raise RuntimeError("negative query_start")

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
<!-- /03-CODE-COMPLETE: attention-final -->


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

<!-- 03-CODE-COMPLETE: llm-engine -->

### 截至 Step 21，`tinyinfer/engine/llm_engine.py` 完整代码

```python
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


class LLMEngine:
    def __init__(
        self,
        config: Config | None = None,
        device: str = "cuda",
    ):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        self.config.load_hf_config()

        # Sequence 的逻辑 block 大小必须与 runtime/cache 一致。
        Sequence.block_size = self.config.kvcache_block_size

        # ModelRunner 先初始化，因为它会根据真实模型显存占用计算
        # config.num_kvcache_blocks；随后 Scheduler/BlockManager 才能使用该值。
        self.model_runner = ModelRunner(self.config, device=device)
        self.scheduler = Scheduler(self.config)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )
        self.config.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else -1
        )
        self.scheduler.eos_token_id = self.config.eos_token_id

    def add_request(
        self,
        token_ids: list[int],
        params: SamplingParams,
    ) -> int:
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        output = self.scheduler.schedule()
        if not output.items:
            return []

        sampled_tokens = self.model_runner.run(output)
        self.scheduler.postprocess(output, sampled_tokens)
        return output.items

    def generate_token_ids(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = list(sampling_params)

        if len(params) != len(prompts):
            raise ValueError("prompts and sampling params length mismatch")

        seqs = []
        for token_ids, p in zip(prompts, params):
            seq = Sequence(token_ids, p)
            seqs.append(seq)
            self.scheduler.add(seq)

        while not self.scheduler.is_finished():
            items = self.step()
            if not items and not self.scheduler.is_finished():
                raise RuntimeError(
                    "scheduler made no progress while requests remain"
                )

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens :],
                "all_token_ids": list(seq.token_ids),
                "num_cached_tokens": seq.num_cached_tokens,
            }
            for seq in seqs
        ]

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
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

### `tinyinfer/llm.py` 完整代码

```python
from tinyinfer.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Public user-facing API。"""

    pass
```
<!-- /03-CODE-COMPLETE: llm-engine -->


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

<!-- 03-CODE-COMPLETE: qwen3-shape-test -->

### `tests/test_qwen3_shapes.py` 完整代码

```python
from types import SimpleNamespace

import torch

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context


def make_fake_qwen3_config():
    return SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        kvcache_block_size=4,
    )


def test_qwen3_mixed_shape_contract_cpu():
    torch.manual_seed(0)
    cfg = make_fake_qwen3_config()

    cache = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=8,
        block_size=cfg.kvcache_block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    model = Qwen3ForCausalLM(cfg, cache).float().eval()

    # 两条 request：第一条 q_len=1，第二条 q_len=3。
    input_ids = torch.tensor([7, 11, 12, 13], dtype=torch.long)
    positions = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    # seq0 使用 physical block 0；seq1 使用 physical block 1。
    set_context(
        q_lens=torch.tensor([1, 3], dtype=torch.int32),
        context_lens=torch.tensor([1, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_seqlen_q=3,
        slot_mapping=torch.tensor([0, 4, 5, 6], dtype=torch.long),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        is_prefill=torch.tensor([False, True]),
        sample_mask=torch.tensor([True, True]),
    )

    try:
        with torch.no_grad():
            hidden = model(input_ids, positions)
            logits = model.compute_logits(hidden[[0, 3]])
    finally:
        reset_context()

    assert hidden.shape == (4, cfg.hidden_size)
    assert logits.shape == (2, cfg.vocab_size)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()
```
<!-- /03-CODE-COMPLETE: qwen3-shape-test -->


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

<!-- 03-CODE-COMPLETE: loader-test -->

### `tests/test_loader.py` 完整代码

```python
import os

import pytest
from transformers import AutoConfig

from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights


@pytest.fixture
def local_qwen3_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path


@pytest.mark.model
def test_loader_loads_every_required_parameter(local_qwen3_path):
    hf_config = AutoConfig.from_pretrained(
        local_qwen3_path,
        trust_remote_code=True,
    )
    hf_config.kvcache_block_size = 16

    # loader accounting 本身不需要真正 KV storage，因此先 kv_cache=None。
    model = Qwen3ForCausalLM(hf_config, kv_cache=None)
    report = load_weights(
        model,
        local_qwen3_path,
        strict=True,
    )

    assert report.missing == []
    assert report.unexpected == []

    print("loaded parameters:", len(report.loaded))
    print("packed shards:", len(report.packed))
    print("skipped tied weights:", report.skipped_tied)
```
<!-- /03-CODE-COMPLETE: loader-test -->


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

<!-- 03-CODE-COMPLETE: hf-logits-test -->

### `tests/test_hf_equivalence.py`：先完成 cold-prefill logits 对齐

```python
import os
import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.context import reset_context, set_context
from tinyinfer.utils.loader import load_weights


@pytest.fixture(scope="module")
def local_model_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path


def _set_single_prefill_context(num_tokens: int, block_size: int):
    num_blocks = math.ceil(num_tokens / block_size)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    set_context(
        q_lens=torch.tensor([num_tokens], dtype=torch.int32),
        context_lens=torch.tensor([num_tokens], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        max_seqlen_q=num_tokens,
        slot_mapping=slot_mapping,
        block_tables=block_table,
        is_prefill=torch.tensor([True]),
        sample_mask=torch.tensor([True]),
    )


@pytest.mark.model
def test_cold_prefill_logits_match_hf(local_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    cfg = hf_model.config
    cfg.kvcache_block_size = 16
    head_dim = int(
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    )

    tiny_model = Qwen3ForCausalLM(cfg, kv_cache=None).to(device, dtype=dtype)
    load_weights(tiny_model, local_model_path, strict=True)

    prompt = "Explain KV cache in one sentence."
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    positions = torch.arange(len(input_ids), dtype=torch.long, device=device)

    num_blocks = math.ceil(len(input_ids) / cfg.kvcache_block_size)
    cache = PagedKVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=max(4, num_blocks),
        block_size=cfg.kvcache_block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    tiny_model.set_kv_cache(cache)
    tiny_model.eval()

    _set_single_prefill_context(len(input_ids), cfg.kvcache_block_size)
    # 把 context tensors 移到模型 device。
    from tinyinfer.utils.context import get_context
    ctx = get_context()
    set_context(
        q_lens=ctx.q_lens.to(device),
        context_lens=ctx.context_lens.to(device),
        cu_seqlens_q=ctx.cu_seqlens_q.to(device),
        max_seqlen_q=ctx.max_seqlen_q,
        slot_mapping=ctx.slot_mapping.to(device),
        block_tables=ctx.block_tables.to(device),
        is_prefill=ctx.is_prefill.to(device),
        sample_mask=ctx.sample_mask.to(device),
    )

    try:
        with torch.no_grad():
            hf_logits = hf_model(ids.unsqueeze(0)).logits[:, -1, :]

            tiny_hidden = tiny_model(ids, positions)
            tiny_logits = tiny_model.compute_logits(tiny_hidden[-1:])
    finally:
        reset_context()

    diff = (tiny_logits.float() - hf_logits.float()).abs()
    print("max_abs_error =", diff.max().item())
    print("mean_abs_error =", diff.mean().item())

    # CPU FP32 可以更严；BF16 baseline 先用较宽阈值，观察后再收紧。
    if dtype == torch.float32:
        torch.testing.assert_close(
            tiny_logits.float(),
            hf_logits.float(),
            atol=1e-4,
            rtol=1e-4,
        )
    else:
        torch.testing.assert_close(
            tiny_logits.float(),
            hf_logits.float(),
            atol=5e-2,
            rtol=5e-2,
        )
```

> 如果这一测试失败，不要先放宽 tolerance。先按 Day06 的逐层排错顺序比较 embedding、norm、Q/K/V、RoPE、attention 与 MLP。
<!-- /03-CODE-COMPLETE: hf-logits-test -->


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

<!-- 03-CODE-COMPLETE: integration-tests -->

### Step 25–28 完整端到端测试文件：`tests/test_engine.py`

下面的测试直接使用公共 `LLM` 接口，因此要求前面的 `ModelRunner + Scheduler + persistent Prefix Cache` 已经全部接通。

```python
import os

import pytest

from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config


@pytest.fixture
def local_model_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path


def make_llm(model_path: str, **overrides):
    values = dict(
        model=model_path,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        max_model_len=512,
        kvcache_block_size=16,
        gpu_memory_utilization=0.80,
    )
    values.update(overrides)
    return LLM(Config(**values))


@pytest.mark.model
def test_greedy_generation_is_deterministic(local_model_path):
    llm = make_llm(local_model_path)
    params = SamplingParams(greedy=True, max_tokens=8)
    prompt = "The capital of France is"

    out1 = llm.generate([prompt], params)[0]["token_ids"]
    out2 = llm.generate([prompt], params)[0]["token_ids"]

    assert out1 == out2


@pytest.mark.model
def test_prefix_cache_does_not_change_output(local_model_path):
    llm = make_llm(local_model_path)
    params = SamplingParams(greedy=True, max_tokens=4)

    shared = (
        "You are a concise assistant. Answer with one short sentence. "
        "Use only factual language.\nQuestion: "
    )
    prompt_a = shared + "What is a KV cache?"
    prompt_b = shared + "What is grouped-query attention?"

    # A 先把 shared prefix 对应 full blocks 变成 persistent cache。
    llm.generate([prompt_a], params)
    cached = llm.generate([prompt_b], params)[0]

    # 新建一个 engine，获得真正 cold baseline。
    cold_llm = make_llm(local_model_path)
    cold = cold_llm.generate([prompt_b], params)[0]

    assert cached["token_ids"] == cold["token_ids"]
    assert cached["num_cached_tokens"] > 0


@pytest.mark.model
def test_mixed_batch_matches_single_request_baseline(local_model_path):
    prompts = [
        "Explain paged KV cache briefly.",
        "What is RoPE?",
        "Why does GQA reduce KV memory?",
    ]
    params = SamplingParams(greedy=True, max_tokens=6)

    # baseline：强制一次只允许一条 request。
    baseline_llm = make_llm(
        local_model_path,
        max_num_seqs=1,
        max_num_batched_tokens=32,
    )
    baseline = [
        baseline_llm.generate([prompt], params)[0]["token_ids"]
        for prompt in prompts
    ]

    # mixed：较小 token budget 强迫 prefill 分 chunk，并允许 decode/prefill 混排。
    mixed_llm = make_llm(
        local_model_path,
        max_num_seqs=8,
        max_num_batched_tokens=12,
    )
    mixed = mixed_llm.generate(prompts, params)
    mixed_tokens = [item["token_ids"] for item in mixed]

    assert mixed_tokens == baseline
```

### Step 25 与 Hugging Face greedy token 对齐：继续加入 `tests/test_hf_equivalence.py`

```python
@pytest.mark.model
def test_first_greedy_token_matches_hf(local_model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        hf_model = hf_model.cuda()
    hf_model.eval()

    prompt = "The capital of France is"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    device = next(hf_model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        hf_next = int(hf_model(input_ids).logits[:, -1, :].argmax(dim=-1))

    tiny = make_llm(local_model_path)
    tiny_next = tiny.generate(
        [prompt],
        SamplingParams(greedy=True, max_tokens=1),
    )[0]["token_ids"][0]

    assert tiny_next == hf_next
```
<!-- /03-CODE-COMPLETE: integration-tests -->


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

<!-- 03-CODE-COMPLETE: sequence-timestamps -->

### `tinyinfer/engine/sequence.py` 完整代码（包含 02-A 状态语义与 Day06 timestamps）

```python
import time
from enum import Enum, auto
from itertools import count

from tinyinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params

        # token_ids 始终包含 prompt + 已采样 completion。
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)

        # 02-A 的三个核心进度。
        self.num_cached_tokens = 0
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0

        # logical block -> physical block id。
        self.block_table: list[int] = []
        self.last_block_hash: int = 0

        # Day06 latency timestamps。
        self.arrival_time = time.perf_counter()
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finished_time: float | None = None
        self.output_token_times: list[float] = []

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        rem = self.num_tokens % self.block_size
        return rem if rem else self.block_size

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

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

    def block_token_ids(self, logical_idx: int) -> list[int]:
        begin = logical_idx * self.block_size
        end = min(begin + self.block_size, self.num_tokens)
        return self.token_ids[begin:end]

    def mark_scheduled(self, n: int) -> None:
        if n <= 0:
            raise ValueError("scheduled token count must be positive")
        if self.num_computed_tokens + n > self.num_tokens:
            raise ValueError("cannot schedule beyond available token ids")
        self.num_scheduled_tokens = n

        if self.first_scheduled_time is None:
            self.first_scheduled_time = time.perf_counter()

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(int(token_id))

        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.output_token_times.append(now)

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.finished_time = time.perf_counter()

    def should_stop(self, eos_token_id: int, max_model_len: int) -> bool:
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True
        if self.num_tokens >= max_model_len:
            return True
        if (
            not self.sampling_params.ignore_eos
            and eos_token_id >= 0
            and self.last_token == eos_token_id
        ):
            return True
        return False
```
<!-- /03-CODE-COMPLETE: sequence-timestamps -->


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

<!-- 03-CODE-COMPLETE: workloads -->

### `benchmarks/workloads.py` 完整代码

```python
from dataclasses import dataclass


@dataclass(slots=True)
class RequestSpec:
    prompt: str
    max_tokens: int


SHORT = [
    RequestSpec(
        prompt="Explain KV cache in one concise paragraph.",
        max_tokens=32,
    ),
    RequestSpec(
        prompt="Explain grouped-query attention in one concise paragraph.",
        max_tokens=32,
    ),
]


# 通过重复固定文本得到 deterministic 长 prefill；不要依赖随机字符串。
_LONG_CONTEXT = (
    "Transformer inference repeatedly applies attention and MLP layers. "
    "Paged KV cache stores historical keys and values in physical blocks. "
) * 180

LONG_PREFILL = [
    RequestSpec(
        prompt=_LONG_CONTEXT + "\nSummarize the paragraph above.",
        max_tokens=32,
    )
]


_SHARED_SYSTEM = (
    "You are a concise technical assistant. "
    "Answer using precise terminology and no bullet points. "
) * 80

SHARED_PREFIX = [
    RequestSpec(
        prompt=_SHARED_SYSTEM + "\nQuestion: What is RoPE?",
        max_tokens=32,
    ),
    RequestSpec(
        prompt=_SHARED_SYSTEM + "\nQuestion: What is GQA?",
        max_tokens=32,
    ),
    RequestSpec(
        prompt=_SHARED_SYSTEM + "\nQuestion: What is chunked prefill?",
        max_tokens=32,
    ),
]


WORKLOADS = {
    "short": SHORT,
    "long_prefill": LONG_PREFILL,
    "shared_prefix": SHARED_PREFIX,
}
```
<!-- /03-CODE-COMPLETE: workloads -->


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

<!-- 03-CODE-COMPLETE: throughput-bench -->

### `benchmarks/benchmark_throughput.py` 完整代码

```python
import argparse
import time

from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config

from benchmarks.workloads import WORKLOADS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workload", choices=WORKLOADS, default="short")
    p.add_argument("--max-num-batched-tokens", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    specs = WORKLOADS[args.workload]

    llm = LLM(
        Config(
            model=args.model,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
        )
    )

    prompts = [x.prompt for x in specs]
    params = [
        SamplingParams(greedy=True, max_tokens=x.max_tokens)
        for x in specs
    ]

    prompt_tokens_total = sum(
        len(llm.tokenizer.encode(p, add_special_tokens=False))
        for p in prompts
    )

    begin = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall_time = time.perf_counter() - begin

    output_tokens_total = sum(len(x["token_ids"]) for x in outputs)
    cached_prompt_tokens = sum(x.get("num_cached_tokens", 0) for x in outputs)
    actual_computed_prompt_tokens = prompt_tokens_total - cached_prompt_tokens

    print(f"wall_time_s={wall_time:.6f}")
    print(f"requests_total={len(outputs)}")
    print(f"prompt_tokens_total={prompt_tokens_total}")
    print(f"cached_prompt_tokens={cached_prompt_tokens}")
    print(f"actual_computed_prompt_tokens={actual_computed_prompt_tokens}")
    print(f"output_tokens_total={output_tokens_total}")
    print(f"output_tokens_per_s={output_tokens_total / wall_time:.3f}")
    print(
        "computed_plus_output_tokens_per_s="
        f"{(actual_computed_prompt_tokens + output_tokens_total) / wall_time:.3f}"
    )
    print(f"requests_per_s={len(outputs) / wall_time:.3f}")


if __name__ == "__main__":
    main()
```
<!-- /03-CODE-COMPLETE: throughput-bench -->


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

<!-- 03-CODE-COMPLETE: latency-bench -->

### `benchmarks/benchmark_latency.py` 完整代码

```python
import argparse
import math
import statistics

from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config

from benchmarks.workloads import WORKLOADS


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile p must be in [0, 100]")

    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]

    rank = (p / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def report(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: "
        f"p50={statistics.median(values) * 1000:.3f} ms, "
        f"p90={percentile(values, 90) * 1000:.3f} ms, "
        f"p99={percentile(values, 99) * 1000:.3f} ms"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workload", choices=WORKLOADS, default="short")
    return p.parse_args()


def main():
    args = parse_args()
    specs = WORKLOADS[args.workload]
    llm = LLM(Config(model=args.model))

    params = [
        SamplingParams(greedy=True, max_tokens=x.max_tokens)
        for x in specs
    ]
    llm.generate([x.prompt for x in specs], params)

    # generate() 完成后，最近一轮 Sequence 已结束；
    # 教学实现建议在 LLMEngine 中额外保留 completed_sequences。
    completed = getattr(llm, "completed_sequences", None)
    if completed is None:
        raise RuntimeError(
            "LLMEngine must expose completed_sequences for latency benchmark"
        )

    ttft = []
    e2e = []
    itl = []

    for seq in completed:
        if seq.first_token_time is not None:
            ttft.append(seq.first_token_time - seq.arrival_time)
        if seq.finished_time is not None:
            e2e.append(seq.finished_time - seq.arrival_time)
        itl.extend(
            b - a
            for a, b in zip(seq.output_token_times, seq.output_token_times[1:])
        )

    report("TTFT", ttft)
    report("ITL", itl)
    report("E2E", e2e)


if __name__ == "__main__":
    main()
```

为了支持上面的 benchmark，在 `LLMEngine.__init__()` 加：

```python
self.completed_sequences = []
```

在每轮 `step()` 的 `postprocess()` 之后，把本轮新结束且尚未记录的 sequence 追加进去。最简单的完整逻辑是维护一个 `self._completed_ids: set[int]`，避免重复加入。
<!-- /03-CODE-COMPLETE: latency-bench -->


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

<!-- 03-CODE-COMPLETE: runtime-metrics -->

### 建议新建 `tinyinfer/utils/metrics.py`，完整代码

```python
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeMetrics:
    prompt_tokens_total: int = 0
    prefix_cached_tokens_total: int = 0

    scheduler_steps: int = 0
    scheduled_tokens_total: int = 0
    decode_tokens_total: int = 0
    prefill_tokens_total: int = 0

    mixed_steps: int = 0
    prefill_only_steps: int = 0
    decode_only_steps: int = 0

    def record_admission(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.prompt_tokens_total += int(prompt_tokens)
        self.prefix_cached_tokens_total += int(cached_tokens)

    def record_schedule(self, output, max_num_batched_tokens: int) -> None:
        del max_num_batched_tokens  # 单步 utilization 在外部按需读取。
        self.scheduler_steps += 1

        prefill = sum(x.num_tokens for x in output.items if x.is_prefill)
        decode = sum(x.num_tokens for x in output.items if not x.is_prefill)

        self.prefill_tokens_total += prefill
        self.decode_tokens_total += decode
        self.scheduled_tokens_total += prefill + decode

        if prefill and decode:
            self.mixed_steps += 1
        elif prefill:
            self.prefill_only_steps += 1
        elif decode:
            self.decode_only_steps += 1

    @property
    def prefix_cache_token_hit_rate(self) -> float:
        if self.prompt_tokens_total == 0:
            return 0.0
        return self.prefix_cached_tokens_total / self.prompt_tokens_total

    @property
    def mixed_step_ratio(self) -> float:
        if self.scheduler_steps == 0:
            return 0.0
        return self.mixed_steps / self.scheduler_steps
```
<!-- /03-CODE-COMPLETE: runtime-metrics -->


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

<!-- 03-CODE-COMPLETE: scheduler-metrics -->

### Scheduler 单步记录代码

在 `Scheduler.schedule()` 返回 `SchedulerOutput(items)` 前加入下面完整逻辑：

```python
output = SchedulerOutput(items)

if hasattr(self, "metrics") and self.metrics is not None:
    self.metrics.record_schedule(
        output,
        self.max_num_batched_tokens,
    )

return output
```

并在 `Scheduler.__init__()` 中接收/创建 `RuntimeMetrics`：

```python
from tinyinfer.utils.metrics import RuntimeMetrics

self.metrics = RuntimeMetrics()
```

单轮 budget utilization 可以直接计算：

```python
utilization = output.num_scheduled_tokens / self.max_num_batched_tokens
```
<!-- /03-CODE-COMPLETE: scheduler-metrics -->


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

<!-- 03-CODE-COMPLETE: invariants -->

### 新建 `tinyinfer/utils/invariants.py`，完整代码

```python
def check_sequence(seq) -> None:
    assert 0 <= seq.num_cached_tokens <= seq.num_prompt_tokens
    assert 0 <= seq.num_computed_tokens <= seq.num_tokens
    assert 0 <= seq.num_scheduled_tokens

    if not seq.prompt_computed:
        assert seq.num_computed_tokens <= seq.num_prompt_tokens
    else:
        assert seq.num_computed_tokens >= seq.num_prompt_tokens
        assert seq.num_tokens - seq.num_computed_tokens in (0, 1)


def check_scheduler_output(output, max_num_batched_tokens: int, max_num_seqs: int) -> None:
    assert output.num_scheduled_tokens <= max_num_batched_tokens
    assert len(output.items) <= max_num_seqs

    for item in output.items:
        assert item.num_tokens > 0
        assert item.start_pos >= 0
        assert item.start_pos + item.num_tokens <= item.seq.num_tokens

        # schedule snapshot 建立时，item.start_pos 就应该等于当时 KV frontier。
        # 如果 schedule() 之后尚未执行 postprocess，这条应始终成立。
        assert item.start_pos == item.seq.num_computed_tokens


def check_block_manager(block_manager) -> None:
    """检查 persistent cache + free-LRU 的双向索引一致性。"""
    free_set = set(block_manager.free_lru.keys())

    for block in block_manager.blocks:
        if block.ref_count == 0:
            assert block.block_id in free_set
        else:
            assert block.block_id not in free_set

        # hash=None 可能是 empty block，也可能是 active partial block；
        # 因此不能要求 token_ids == ()。
        assert len(block.token_ids) <= block_manager.block_size

        if block.hash is not None:
            # 只有 full block 才允许进入 Prefix Cache 索引。
            assert len(block.token_ids) == block_manager.block_size
            ids = block_manager.hash_to_block_ids.get(block.hash, set())
            assert block.block_id in ids

    for block_hash, ids in block_manager.hash_to_block_ids.items():
        assert ids
        for block_id in ids:
            block = block_manager.blocks[block_id]
            assert block.hash == block_hash
```
<!-- /03-CODE-COMPLETE: invariants -->


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

<!-- 03-CODE-COMPLETE: trace -->

### `tinyinfer/utils/debug.py` 完整代码

```python
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    ts: float
    name: str
    seq_id: int | None = None
    fields: dict[str, Any] = field(default_factory=dict)


class TraceWriter:
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def emit(self, name: str, seq_id: int | None = None, **fields) -> None:
        if not self.enabled:
            return

        event = TraceEvent(
            ts=time.perf_counter(),
            name=name,
            seq_id=seq_id,
            fields=fields,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def trace_enabled(name: str) -> bool:
    return os.getenv(name, "0") == "1"


def make_default_trace_writer() -> TraceWriter:
    path = os.getenv("TINYINFER_TRACE_FILE")
    return TraceWriter(path)
```

使用示例：

```python
self.trace.emit("schedule", num_items=len(output.items), tokens=output.num_scheduled_tokens)
self.trace.emit("block_evict", block_id=block_id, old_hash=old_hash)
self.trace.emit("sample", seq_id=seq.seq_id, token_id=token_id)
```
<!-- /03-CODE-COMPLETE: trace -->


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

<!-- 03-CODE-COMPLETE: debug-correctness -->

### `debug_correctness` 完整 runtime 应用 helper

建议在 `ModelRunner.__init__()` 最前面调用：

```python
def apply_debug_correctness_overrides(config) -> None:
    if not config.debug_correctness:
        return

    # correctness 模式禁止任何会改变执行路径、引入异步状态的性能优化。
    config.enforce_eager = True
```

然后测试时：

```python
cfg = Config(
    model=model_path,
    debug_correctness=True,
)
```

`Sampler` 不需要全局强制 greedy；correctness 测试显式使用：

```python
SamplingParams(greedy=True, max_tokens=8)
```
<!-- /03-CODE-COMPLETE: debug-correctness -->


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

<!-- 03-CODE-COMPLETE: conftest -->

### `tests/conftest.py` 完整代码

```python
import os

import pytest


@pytest.fixture(scope="session")
def local_model_path():
    path = os.getenv("TINYINFER_MODEL")
    if not path:
        pytest.skip("TINYINFER_MODEL is not set")
    return path
```

`pyproject.toml` 中确保有：

```toml
[tool.pytest.ini_options]
markers = [
    "model: requires a local Hugging Face model checkpoint",
]
```
<!-- /03-CODE-COMPLETE: conftest -->


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
