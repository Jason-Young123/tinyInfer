# tinyInfer 精度对齐问题复盘 Note1

## 1. RoPE 精度问题：`register_buffer`、`Module._apply` 与 FP32 计算

### 1.1 RoPE 中的 `inv_freq` 是什么？

RoPE 通过旋转位置编码：

$$
\theta = position \times inv\_freq
$$

其中：

- `position`：token 的绝对位置
- `inv_freq`：不同维度对应的旋转频率

典型实现：

```python
inv_freq = 1.0 / (
    base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        / rotary_dim
    )
)
```

之后：

```text
position
   ↓
× inv_freq
   ↓
freqs
   ↓
cos / sin
   ↓
旋转 Q / K
```

---

### 1.2 为什么 `inv_freq` 适合使用 `register_buffer`？

PyTorch 中常见的状态可以分成三类。

#### Parameter

```python
self.weight = nn.Parameter(...)
```

特点：

- 是模型参数
- 可参与梯度更新
- 会跟随 `model.to(...)` 迁移 device / dtype
- 会保存到 `state_dict`

#### Buffer

```python
self.register_buffer("inv_freq", tensor)
```

特点：

- 不是可训练参数
- 会跟随 `model.to(...)` 迁移
- 默认会进入 `state_dict`

RoPE 中通常使用：

```python
self.register_buffer(
    "inv_freq",
    inv_freq,
    persistent=False,
)
```

`persistent=False` 表示它不写入 `state_dict`，因为它可以根据 `base` 和 `rotary_dim` 重新计算。

#### 普通成员变量

```python
self.inv_freq = tensor
```

PyTorch 不会自动管理它：

```python
model.to("cuda")
```

不会自动把这个 tensor 搬到 CUDA。

因此，`inv_freq` 的特点是：

- 不需要训练
- 但参与 forward
- 需要跟随模型迁移 device

所以适合作为 buffer。

---

### 1.3 为什么 `model.to(dtype=torch.bfloat16)` 会影响 `inv_freq`？

执行：

```python
model.to(
    device="cuda",
    dtype=torch.bfloat16,
)
```

内部大致会走：

```text
Module.to()
   ↓
Module._apply(...)
   ↓
递归处理 Parameter / Buffer
```

如果 `inv_freq` 是 buffer，默认也会被转换：

```text
inv_freq: FP32
      ↓
model.to(dtype=BF16)
      ↓
inv_freq: BF16
```

问题在于 RoPE 频率属于数值敏感的数学常量。

一旦发生：

```text
FP32 → BF16
```

就已经产生了量化误差。后续即使：

```python
inv_freq.float()
```

也只能得到：

```text
BF16 量化后的数值 → FP32
```

无法恢复原始 FP32 精度。

---

### 1.4 为什么重写 `_apply()` 可以解决？

`nn.Module.to()` 内部会调用模块的 `_apply()`。

因此可以在 `RotaryEmbedding` 中重写：

```python
def _apply(self, fn, recurse=True):
    inv_freq_fp32 = self.inv_freq

    super()._apply(fn, recurse)

    self.inv_freq = inv_freq_fp32.to(
        device=self.inv_freq.device
    )

    return self
```

逻辑：

```text
先保存原始 FP32 inv_freq
        ↓
让 PyTorch 正常执行 model.to(...)
        ↓
Parameter / Buffer 正常迁移
        ↓
重新用原始 FP32 inv_freq 覆盖回来
```

最终：

```text
Linear weight → BF16 CUDA
inv_freq      → FP32 CUDA
```

注意：

> 这里重写的是 `RotaryEmbedding._apply()`，不是 `inv_freq._apply()`。  
> `inv_freq` 本身只是一个 Tensor。

---

### 1.5 为什么 forward 里仍然写 `dtype=torch.float32`？

即使 `_apply()` 已经保证正常路径下 `self.inv_freq` 是 FP32，forward 中仍推荐：

```python
inv_freq = self.inv_freq.to(
    device=q.device,
    dtype=torch.float32,
)
```

这是两层不同的保护：

- `_apply()`：保证保存下来的 buffer 状态不会被 BF16 污染
- forward：保证本次 RoPE 数学计算使用 FP32

因此推荐：

```text
inv_freq  → FP32
positions → FP32
freqs     → FP32
cos / sin → FP32 生成
```

最后再根据模型实现，将 `cos/sin` 或旋转结果转换到 Q/K 的 dtype。

---

### 1.6 为什么 forward 里还要写 `.to(device=q.device)`？

正常情况下：

```python
model.to("cuda")
```

已经会把 buffer 搬到 CUDA，因此这一步通常不会真的发生数据拷贝。

但它可以保证模块独立调用时仍然安全，例如：

```python
rope = RotaryEmbedding(...)
q = q.cuda()
rope(q, k, positions)
```

即使整个 `rope` 没有提前 `.cuda()`，forward 也能保证 `inv_freq` 与输入位于同一 device。

因此：

```python
.to(
    device=q.device,
    dtype=torch.float32,
)
```

同时明确：

1. 本次计算和 Q/K 使用同一 device
2. RoPE 频率计算必须使用 FP32

---

## 2. QKV 拼接矩阵乘法为什么会产生数值差异？

### 2.1 数学上完全等价

HuggingFace 常见写法：

$$
Q = XW_q^T
$$

$$
K = XW_k^T
$$

$$
V = XW_v^T
$$

tinyInfer 可以把权重沿输出维拼接：

$$
W_{qkv}
=
\begin{bmatrix}
W_q \\
W_k \\
W_v
\end{bmatrix}
$$

然后一次计算：

$$
XW_{qkv}^T = [Q,K,V]
$$

所以从线性代数角度：

> 三次独立 GEMM 和一次 packed GEMM 完全等价。

---

### 2.2 为什么实际 GPU 上结果可能不同？

原因不是 Q/K/V 互相影响，而是：

> 整个 GEMM 的 shape 改变后，底层 kernel 的执行方式可能改变。

例如 HF：

```text
GEMM 1: X @ Wq.T
GEMM 2: X @ Wk.T
GEMM 3: X @ Wv.T
```

packed：

```text
GEMM: X @ Wqkv.T
```

GPU 会根据矩阵尺寸决定：

- tile 大小
- warp 分配
- Tensor Core 使用方式
- reduction tree
- 浮点累加顺序

浮点加法不满足严格结合律：

$$
(a+b)+c \neq a+(b+c)
$$

所以即使某一列使用的仍然是完全相同的 `Wq/Wk/Wv` 数据，只要累加顺序变化，最后几个 bit 就可能不同。

关键理解：

> 不是 Wk/Wv 数值影响了 Wq，而是“整体矩阵 shape 改变”导致 Wq 自己的计算路径可能改变。

---

### 2.3 为什么 tinyInfer 仍然值得使用 packed QKV？

工程推理更偏向 packed QKV：

```text
3 GEMM
  ↓
1 GEMM
```

优点：

- 减少 kernel launch
- 改善内存访问
- 更接近高性能推理框架的实现

但如果目标是和 HuggingFace 做逐元素甚至 bitwise 对齐，可以暂时改成：

```python
q = F.linear(x, Wq)
k = F.linear(x, Wk)
v = F.linear(x, Wv)
qkv = torch.cat((q, k, v), dim=-1)
```

权重仍然可以保持 packed 存储，loader 不需要修改，只改变 forward 的计算路径。

因此：

> packed QKV 是性能优化，不是数学错误；HF equivalence 测试和生产推理可以采用不同计算模式。

---

## 3. Attention：SDPA 与 Eager

### 3.1 Eager Attention

经典 attention：

$$
Attention(Q,K,V)
=
softmax\left(
\frac{QK^T}{\sqrt{d}} + Mask
\right)V
$$

Eager 实现通常显式执行：

```text
Q @ K^T
   ↓
加 causal mask
   ↓
FP32 softmax
   ↓
转回模型 dtype
   ↓
@ V
```

优点：

- 数学过程直观
- 适合 reference/debug

缺点：

- 中间矩阵大
- kernel 较多
- 性能一般不如融合实现

---

### 3.2 SDPA 是什么？

PyTorch 提供：

```python
torch.nn.functional.scaled_dot_product_attention
```

简称 SDPA。

它表达的数学公式仍然是 scaled dot-product attention，但底层可以选择优化 backend，例如：

- Flash Attention
- memory-efficient attention
- math backend

因此可以把多步操作融合起来，减少显存和 kernel 开销。

---

### 3.3 为什么 SDPA 与 Eager 数值不一定完全一致？

原因和 packed QKV 类似：

> 数学相同，但浮点执行路径不同。

Eager：

```text
matmul
→ softmax
→ matmul
```

SDPA：

```text
可能由融合 kernel 一次完成
```

因此：

- 累加顺序不同
- 中间 dtype / rounding 路径可能不同
- BF16 下更容易出现逐元素差异

这不代表其中一方实现错误。

---

### 3.4 tinyInfer 应该对齐哪个？

如果 tinyInfer 使用：

```python
F.scaled_dot_product_attention(...)
```

那么 HF reference 测试最好显式：

```python
AutoModelForCausalLM.from_pretrained(
    path,
    attn_implementation="sdpa",
)
```

这样比较的是：

```text
HF SDPA
vs
tinyInfer SDPA
```

如果想验证最直观的 reference arithmetic，则两边都应该使用 eager。

原则：

> 数值对齐测试必须先保证 reference model 和 tinyInfer 使用同一种 attention backend。

---

## 4. LM Head：为什么一次矩阵乘法能得到整个词表的分数？

### 4.1 LM Head 本质是线性分类器

Transformer 最终 hidden state：

$$
H \in \mathbb{R}^{T \times d}
$$

LM Head 权重：

$$
W \in \mathbb{R}^{V \times d}
$$

其中：

- `T`：token 数
- `d`：hidden size
- `V`：vocab size

PyTorch `nn.Linear(d, V)` 实际计算：

$$
Logits = HW^T
$$

shape：

$$
[T,d] \times [d,V]
\rightarrow
[T,V]
$$

因此每一行直接得到整个 vocabulary 的所有 logits。

---

### 4.2 为什么一次矩阵乘法等价于对每个 token 做内积？

第 $i$ 个 vocabulary token 的输出：

$$
logit_i = h \cdot W_i
$$

也就是当前 hidden vector 与 `W` 第 `i` 行的点积。

矩阵乘法只是把：

```text
与 vocab token 0 点积
与 vocab token 1 点积
...
与 vocab token V-1 点积
```

一次性并行完成。

所以可以把 logit 理解为：

> 当前上下文 hidden state 与某个候选 token 输出方向的 compatibility score。

---

### 4.3 为什么不用 cosine similarity？

余弦相似度：

$$
\cos(h,w_i)
=
\frac{h\cdot w_i}
{\|h\|\|w_i\|}
$$

它只保留方向信息。

但 LM Head 本质不是 embedding retrieval，而是：

> 一个训练出来的 $V$ 类线性分类器。

直接点积：

$$
h\cdot w_i
$$

不仅包含方向，还保留向量尺度。

模型可以利用：

- 权重方向
- 权重范数
- hidden state 的尺度

共同控制 logits 和 softmax 的尖锐程度。

如果推理时额外归一化，反而会改变训练时学习到的决策函数。

因此主流 causal LM 通常直接使用线性投影，而不是 cosine classifier。

---

## 5. 为什么 Embedding 与 LM Head 可以共享同一个 Weight？

### 5.1 Embedding 查表

Embedding 权重：

$$
E \in \mathbb{R}^{V \times d}
$$

输入 token id 为 `i` 时：

```python
hidden = E[i]
```

本质是从矩阵中取第 `i` 行：

```text
token id
   ↓
查表
   ↓
对应 token vector
```

---

### 5.2 LM Head 使用同一个矩阵

如果使用 weight tying：

```python
self.lm_head.weight = self.model.embed_tokens.weight
```

那么：

$$
W_{lm} = E
$$

LM Head：

$$
logits = HE^T
$$

因此同一个矩阵有两个方向的用途：

```text
输入阶段：

token id
   ↓
E[token_id]
   ↓
hidden vector


输出阶段：

hidden vector
   ↓
hidden @ E.T
   ↓
vocab logits
```

---

### 5.3 为什么这种共享合理？

输入 embedding 的任务：

> 把离散 token 映射到 hidden space。

输出 LM Head 的任务：

> 根据当前 hidden state 判断哪个 token 最匹配。

二者天然围绕同一个 token 语义空间工作，因此共享矩阵是合理的。

优点：

- 减少参数量
- 输入/输出 token 空间保持一致
- 实践中通常具有良好效果

真正的 weight tying 是：

```python
lm_head.weight is embed_tokens.weight
# True
```

即两个 module 引用同一个 `nn.Parameter` 和同一块 storage，而不是复制两份相同数值。

---

## 6. `atol`、`rtol` 与误差判断公式

### 6.1 `torch.testing.assert_close`

常见判断规则：

$$
|x-y|
\le
atol + rtol \times |y|
$$

其中通常：

- `x`：待测试结果，例如 tinyInfer 输出
- `y`：reference，例如 HuggingFace 输出

---

### 6.2 `atol` 是什么？

`atol`：

> absolute tolerance，绝对容忍误差。

例如：

```python
atol = 1e-4
```

意味着即使 reference 很接近 0，也至少允许大约：

$$
10^{-4}
$$

的绝对差异。

它尤其适合处理 reference 接近 0 的情况，因为此时相对误差会变得非常大甚至失去意义。

---

### 6.3 `rtol` 是什么？

`rtol`：

> relative tolerance，相对容忍误差。

它允许容差随 reference 数值大小增加：

$$
rtol \times |y|
$$

例如：

```text
y = 100
rtol = 0.001
```

相对容差为：

$$
0.001 \times 100 = 0.1
$$

因此大数允许更大的绝对误差。

---

### 6.4 为什么同时需要 `atol` 和 `rtol`？

只使用相对误差：

- 当 `y ≈ 0` 时，相对误差容易爆炸

只使用绝对误差：

- 当 `y` 很大时，同一个固定阈值可能过于严格

所以组合：

$$
|x-y|
\le
atol + rtol|y|
$$

同时兼顾：

```text
接近 0 的数 → atol 主导
大数        → rtol 主导
```

例如：

```text
y = 100
atol = 0.01
rtol = 0.001
```

允许误差：

$$
0.01 + 0.001 \times 100 = 0.11
$$

如果：

```text
x = 100.08
```

那么：

$$
|100.08 - 100| = 0.08 < 0.11
$$

因此认为两者足够接近。
