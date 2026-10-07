# note2_sampling.md

# Sampling：Softmax / Exponential Race / Gumbel-Max

## 1. 统一符号

假设模型已经得到单条请求的下一 token logits：

$$
z \in \mathbb{R}^{V}
$$

其中：

- $V$：vocab size
- $z_i$：第 $i$ 个 token 的 logit
- $T > 0$：temperature

先做 temperature scaling：

$$
s_i = \frac{z_i}{T}
$$

即：

```text
logits [V]
   ↓ / T
scaled_logits [V]
```

temperature 作用：

```text
T < 1  → 分布更尖锐，更确定
T = 1  → 原始分布
T > 1  → 分布更平缓，更随机
```

> 采样阶段处理的是 vocab 维度，因此这里的向量长度应为 vocab size $V$，不是 hidden dim $d$。hidden dim $d$ 属于前面的 Transformer hidden state；经过 LM Head 后才得到 logits `[V]`。

---

# 2. 方法一：Softmax + Multinomial

## 2.1 完整流程

输入：

$$
z \in \mathbb{R}^{V}
$$

### Step 1：Temperature scaling

$$
s_i = \frac{z_i}{T}
$$

```text
z [V]
 ↓
s [V]
```

### Step 2：Softmax

$$
p_i =
\frac{e^{s_i}}
{\sum_{j=1}^{V} e^{s_j}}
$$

得到：

$$
p \in \mathbb{R}^{V}
$$

满足：

$$
p_i \ge 0
$$

且：

$$
\sum_i p_i = 1
$$

### Step 3：Categorical sampling

按照概率 $p_i$ 抽取 token：

$$
k \sim \mathrm{Categorical}(p)
$$

PyTorch：

```python
scaled = logits / temperature
probs = torch.softmax(scaled.float(), dim=-1)
token = torch.multinomial(probs, num_samples=1)
```

最终：

```text
logits [V]
   ↓ / T
scaled [V]
   ↓ softmax
probs [V]
   ↓ multinomial
token_id
```

---

# 3. 方法二：Exponential Race

核心思想：

> 为每个 token 构造一个随机“到达时间”，谁最早到达，谁被采样。

## 3.1 完整流程

输入：

$$
z \in \mathbb{R}^{V}
$$

### Step 1：Temperature scaling

$$
s_i = \frac{z_i}{T}
$$

### Step 2：转成正权重

$$
w_i = e^{s_i}
$$

不需要归一化，因为：

$$
p_i =
\frac{w_i}
{\sum_j w_j}
$$

### Step 3：生成指数随机变量

对每个 token：

$$
E_i \sim \mathrm{Exp}(1)
$$

也可以从：

$$
U_i \sim \mathrm{Uniform}(0,1)
$$

生成：

$$
E_i = -\log U_i
$$

### Step 4：计算 race time

$$
\tau_i =
\frac{E_i}{w_i}
$$

### Step 5：选择最早到达者

$$
k =
\arg\min_i \tau_i
$$

即：

$$
k =
\arg\min_i
\frac{E_i}{e^{s_i}}
$$

等价地：

$$
k =
\arg\max_i
\frac{e^{s_i}}{E_i}
$$

完整数据流：

```text
logits [V]
   ↓ / T
scaled [V]
   ↓ exp
weights [V]
   ↓
E_i ~ Exp(1)
   ↓
E_i / weights_i
   ↓
argmin
   ↓
token_id
```

最终：

$$
P(k=i)
=
\frac{e^{s_i}}
{\sum_j e^{s_j}}
$$

因此与 Softmax + Multinomial 完全等价。

---

# 4. 方法三：Gumbel-Max

Gumbel-Max 可以看作 Exponential Race 的 log-domain 形式。

## 4.1 完整流程

输入：

$$
z \in \mathbb{R}^{V}
$$

### Step 1：Temperature scaling

$$
s_i = \frac{z_i}{T}
$$

### Step 2：生成 Gumbel noise

先生成：

$$
U_i \sim \mathrm{Uniform}(0,1)
$$

然后：

$$
G_i =
-\log\left(-\log U_i\right)
$$

于是：

$$
G_i \sim \mathrm{Gumbel}(0,1)
$$

### Step 3：加到 scaled logits

$$
y_i = s_i + G_i
$$

### Step 4：Argmax

$$
k =
\arg\max_i \left(s_i + G_i\right)
$$

PyTorch：

```python
scaled = logits / temperature
u = torch.rand_like(scaled)
g = -torch.log(-torch.log(u))
token = (scaled + g).argmax(dim=-1)
```

完整数据流：

```text
logits [V]
   ↓ / T
scaled [V]
   +
Gumbel noise [V]
   ↓
argmax
   ↓
token_id
```

最终：

$$
P(k=i)
=
\frac{e^{s_i}}
{\sum_j e^{s_j}}
$$

---

# 5. Exponential Race 与 Gumbel-Max 的等价关系

Exponential Race：

$$
k =
\arg\min_i
\frac{E_i}{e^{s_i}}
$$

改写为：

$$
k =
\arg\max_i
\frac{e^{s_i}}{E_i}
$$

因为 $\log(\cdot)$ 单调递增，所以取 log 不改变 argmax：

$$
k =
\arg\max_i
\left(
s_i - \log E_i
\right)
$$

又因为：

$$
E_i \sim \mathrm{Exp}(1)
$$

且：

$$
G_i = -\log E_i
$$

满足：

$$
G_i \sim \mathrm{Gumbel}(0,1)
$$

因此：

$$
k =
\arg\max_i
\left(
s_i + G_i
\right)
$$

即：

```text
Exponential Race
      ↓ 取 log
Gumbel-Max
```

两者只是不同的数学表示。

---

# 6. 三种方法的统一关系

三者最终采样的目标分布完全相同：

$$
P(k=i)
=
\mathrm{Softmax}\left(\frac{z}{T}\right)_i
$$

关系：

```text
                logits z
                   │
                   ▼
                z / T
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
     softmax      exp      + Gumbel
        │          │          │
        ▼          ▼          ▼
 multinomial   Exp Race     argmax
        │          │          │
        └──────────┼──────────┘
                   ▼
             sampled token
```

因此：

$$
\mathrm{Softmax + Multinomial}
\equiv
\mathrm{Exponential\ Race}
\equiv
\mathrm{Gumbel\text{-}Max}
$$

这里的“等价”指：

> 三者产生完全相同的 categorical sampling distribution。

单次随机结果不必相同。

---

# 7. 执行效率对比

设 vocab size 为 $V$。

三种方法的渐进复杂度均为：

$$
O(V)
$$

因为最终都必须处理整个 vocab。

| 方法 | 主要操作 | 是否显式 Softmax | 随机数数量 | 理论复杂度 | GPU 实现特点 |
|---|---|---:|---:|---:|---|
| Softmax + Multinomial | softmax + categorical sampling | 是 | 少量 / 实现相关 | $O(V)$ | PyTorch 内置，简单可靠 |
| Exponential Race | exp + Exp RNG + division + argmin | 否 | $V$ | $O(V)$ | 适合与 reduction 融合 |
| Gumbel-Max | Gumbel RNG + add + argmax | 否 | $V$ | $O(V)$ | 非常适合 fused CUDA reduction |

---

# 8. 实际性能判断

## Softmax + Multinomial

优点：

```text
实现最简单
PyTorch 内置
语义清晰
数值和工程行为成熟
```

缺点：

```text
需要显式 softmax
通常需要 materialize probs[V]
之后再执行 sampling
```

---

## Exponential Race

优点：

```text
不需要概率归一化
数学上直接完成 categorical sampling
适合融合实现
```

缺点：

```text
naive 实现需要 exp(logits)
还要生成 V 个随机数
```

---

## Gumbel-Max

优点：

```text
直接工作在 logits 上
不需要 softmax
不需要 exp(logits)
最终只需 reduction argmax
适合 fused CUDA kernel
```

缺点：

```text
需要为每个 vocab token 生成随机噪声
naive PyTorch 写法包含 rand/log/log/add/argmax 多个 kernel
未融合时不一定比 torch.multinomial 快
```

---

# 9. 工程结论

对于教学版 PyTorch sampler：

```python
scaled = logits / temperature
probs = torch.softmax(scaled.float(), dim=-1)
token = torch.multinomial(probs, 1)
```

通常最合适，因为：

```text
简单
清晰
可靠
```

对于高性能 CUDA sampler：

```text
优先考虑：

Gumbel-Max
或
Exponential Race
```

原因：

```text
无需显式 softmax
无需 materialize normalized probabilities
更容易与 RNG + reduction 融合
```

但最终性能取决于具体 kernel 实现，而不是仅由算法复杂度决定。

---

# 10. 最简记忆版

```text
Softmax + Multinomial

logits / T
   ↓
softmax
   ↓
概率采样
```

```text
Exponential Race

logits / T
   ↓
exp → weight
   ↓
E ~ Exp(1)
   ↓
argmin(E / weight)
```

```text
Gumbel-Max

logits / T
   ↓
+ Gumbel(0,1)
   ↓
argmax
```

三者最终：

$$
P(\text{token}=i)
=
\mathrm{Softmax}\left(\frac{\text{logits}}{T}\right)_i
$$

并且：

$$
\mathrm{Exponential\ Race}
\Longleftrightarrow
\mathrm{Gumbel\text{-}Max}
$$
