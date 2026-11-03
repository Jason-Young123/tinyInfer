# tinyInfer Advance 07：Speculative Decoding 手把手实现与双 KV Cache 一致性

本讲以本次上传的 `tinyInfer-master(20261027-152019).zip` 为唯一源码基线，沿用 `docs/advance04_flashattention.md`、`docs/advance05_flashdecoding.md` 和 `docs/advance06_memory_profile_cache_hit.md` 的逐步教学结构。本文从模型下载开始，随后给出所有修改/新增文件的**完整文件内容**，不是伪代码，也不要求读者自行补齐省略部分。

> 实现目标：Target 使用本地 `Qwen3-4B`，Draft 使用已有的 `Qwen3-0.6B`；SD 作为显式开关，不启用时沿用原始 tinyInfer 推理路径。保留现有 5 个 User panel、continuous batching / chunked prefill、Paged KV、Prefix Cache、Attention backend 与性能面板。SD 采用单分支候选序列，支持固定最大 draft 长度和可选的简单自适应 draft 长度。

## 本讲完成后你应该能够解释什么

- 为什么 SD 能把多个 Target token 的验证合并进一次 Target forward，以及为什么 Draft 仍然需要自回归地产生候选。
- 论文中的接受概率、拒绝后的 residual distribution 为什么能够保持 Target 的采样分布。
- 为什么 `target_model` 和 `draft_model` 必须拥有**两份独立的 KV tensor storage**，但可以复用同一套逻辑 block ID / `block_table`。
- 一轮验证结束后，如何从 `verified_end` 和 `accepted_count` 算出 `keep_tokens`，并同时回滚 Target 和 Draft 的无效 KV 后缀。
- 为什么回滚通常是恢复逻辑有效前缀，而不是把 token 从 Python list 删除；为什么本讲还会把无效 slot 清零作为教学实现中的防御性措施。
- 为什么统一使用一套 token-prefix hash 和逻辑 Block metadata 就足够；以及如何通过双模型 KV frontier 保证只有 Target 与 Draft 都完整写入的 block 才能注册 Prefix Cache。
- 如何把两个模型的权重、两份 KV cache 和临时激活一起纳入 block 数估算，避免仍按单模型预算导致 OOM。
- 如何在 UI 中动态调整 draft token 上限，并观测 proposal 数、accepted 数、acceptance rate 与每轮平均接受数。
- 为什么“理论上保持 Target 分布”不等于“GPU 低精度实现逐 token bitwise-equivalent”，以及哪些指标才适合判断 SD 是否真的加速。

## 0. 实现边界与重要声明

本讲实现的是一个**可验证、可调试的教学级 SD 引擎**，并非宣称复制了 vLLM、SGLang、TensorRT-LLM 或其他工业引擎的全部实现。具体范围如下。

| 能力 | 本讲实现 | 当前明确不实现 |
|---|---|---|
| Draft 结构 | 单分支候选序列 | Draft tree / tree attention |
| Target 验证 | 多用户 ragged batch，一次 forward 验证每个用户的候选段 | 跨请求 speculative scheduler 的全部生产级优化 |
| 采样 | Greedy；temperature + 与原 Sampler 一致的 top-k=100 分布；拒绝采样 | top-p、repetition penalty、grammar constraint、bad-word processor |
| KV cache | Target/Draft 独立 storage、共享逻辑 block ID 和单一 token-prefix hash；双模型完整 block 才注册；拒绝后缀回滚但保留已预留 block ID | CUDA Graph、专用的 KV copy-on-write、树形 KV 分叉 |
| Draft 长度 | UI 可调最大值；可选 EMA acceptance-rate heuristic | 基于完整硬件代价模型的工业级动态 gamma controller |
| UI | 保留 5 个 User panel，增加 SD 控制和统计 | 逐 token 显示每个接受 token 的独立硬件时间戳 |
| 并行 | 保留现有请求并发和 mixed scheduling；Draft step 与 Target verify 按 ragged batch 执行 | Tensor Parallel / 多 GPU 分布式 SD |

**不要把本讲的 heuristic 描述成工业级最优控制器。** `--adaptive-draft` 使用近期接受率的 EMA 做简单增减，主要用于学习动态 gamma 的工程接口。工业实现通常还会综合 Draft/Target latency、候选长度、接受分布、批大小、GPU 利用率和服务目标进行调优。

本讲的采样分布与原 tinyInfer `Sampler` 保持一致：非 greedy 模式使用 temperature 和 top-k=100；Greedy 不受 temperature 影响。若未来添加 top-p 或其他 logit processor，必须让 Draft proposal、Target 验证分布和 residual correction 遵守同一套采样规则，否则不能直接声称保持原 Target 分布。

## 1. Step 1：下载 Qwen3-4B Target 模型

本地已有模型路径按仓库历史环境约定为：

```bash
/home/jason/huggingface/Qwen3-0.6B
```

新下载 Target 到：

```bash
/home/jason/huggingface/Qwen3-4B
```

官方模型仓库为 [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)，Draft 对应 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。Hugging Face 当前推荐的 CLI 命令是 `hf download ... --local-dir ...`，其官方说明见 [Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cli)。

### 1.1 安装或更新下载工具

在你运行 tinyInfer 的同一个 WSL Ubuntu 环境中执行：

```bash
python -m pip install -U huggingface_hub
hf --help
```

若模型需要登录权限，先执行：

```bash
hf auth login
```

Qwen3 官方公开仓库通常不需要额外授权；若当前网络访问 Hugging Face 不稳定，先解决网络/代理问题，不要用不完整的模型目录启动 tinyInfer。

### 1.2 下载 Target

```bash
mkdir -p /home/jason/huggingface
hf download Qwen/Qwen3-4B \
  --local-dir /home/jason/huggingface/Qwen3-4B
```

Draft 如果已经完整存在，不需要重新下载。若想确保本地目录与官方仓库文件齐全，可执行：

```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir /home/jason/huggingface/Qwen3-0.6B
```

若当前环境只有旧版 CLI，可以尝试：

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir /home/jason/huggingface/Qwen3-4B
```

### 1.3 检查目录结构

```bash
ls -lah /home/jason/huggingface/Qwen3-4B
ls -lah /home/jason/huggingface/Qwen3-0.6B
find /home/jason/huggingface/Qwen3-4B -maxdepth 1 -name '*.safetensors' -printf '%f %s bytes\n'
```

至少要确认模型目录中存在 `config.json`、tokenizer 相关文件，以及一个或多个位于**目录顶层**的 `.safetensors` 权重文件。本仓库的 `iter_safetensor_weights()` 直接遍历 `model_path.glob("*.safetensors")`，不会递归搜索子目录；不要把权重只下载到嵌套子目录中。

### 1.4 显存与磁盘预期

Qwen3-4B 的 BF16 权重约为数 GB 量级，Qwen3-0.6B 还会额外占用一份权重空间；运行时还需要两份 KV pool、激活和 CUDA/PyTorch 临时空间。**不能只用 Target 单模型的显存账本来估算 SD 模式。** 本讲会在两个模型均加载后重新 profile，并按每个逻辑 block 的 Target + Draft 总成本计算可分配 block 数。具体可分配数量以你的 GPU、模型权重 dtype、`max_num_batched_tokens` 和 `gpu_memory_utilization` 为准，不在讲义中预先编造一个固定数字。

## 2. Step 2：理解 SD 的概率算法

原始论文：Yaniv Leviathan、Matan Kalman、Yossi Matias，*Fast Inference from Transformers via Speculative Decoding*，ICML 2023。论文的核心是：Draft 先生成候选，Target 并行计算这些候选位置对应的 next-token 分布，再用修正后的拒绝采样保持 Target 的输出分布。参考：[PMLR 原始论文页面](https://proceedings.mlr.press/v202/leviathan23a.html)。

### 2.1 定义两个分布

对同一个已确认前缀，Draft 的下一个 token 分布记为 q，Target 的下一个 token 分布记为 p。Draft 从 q 中采样候选 token x。对于非 greedy 模式，接受概率为：

$$
P(\text{accept }x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

如果候选被拒绝，不能直接从 p 重新采一个 token 就结束推导；论文要求从 residual distribution 采样：

$$
r(x)=\max(0,p(x)-q(x))
$$

$$
p_{\text{correct}}(x)=\frac{r(x)}{\sum_y r(y)}
$$

当候选被接受时，输出候选 token；被拒绝时，输出从 residual distribution 中抽取的 correction token，并丢弃后续 Draft 候选。在一轮的所有 Draft 候选都被接受后，再从 Target 在最后一个候选之后给出的分布中抽取一个 bonus token。因此一轮最多输出 K+1 个 token，而不是固定只输出 K 个。

### 2.2 为什么拒绝修正能够保持 Target 分布

对于某个 token x，来自“候选被接受”分支的概率质量是：

$$
q(x)\min\left(1,\frac{p(x)}{q(x)}\right)=\min(q(x),p(x))
$$

拒绝事件的总概率为：

$$
P(\text{reject})=1-\sum_x\min(p(x),q(x))=\sum_x\max(p(x)-q(x),0)
$$

因此 correction 分支最终输出某个 token x 的概率质量为：

$$
P(\text{reject})\,p_{\text{correct}}(x)=\max(p(x)-q(x),0)
$$

接受分支贡献 `min(p(x),q(x))`，拒绝分支贡献 `max(p(x)-q(x),0)`；两者相加就是 p(x)。因此在精确算术下，输出 token 的边缘分布仍然是 Target 的 p，而不是 Draft 的 q。

这里有三个不能混淆的点：

1. **接受率高不等于正确性更高。** 正确性由接受/拒绝修正规则保证；接受率主要影响性能。
2. **不能只比较 logits 大小。** 非 greedy SD 需要 p(x)、q(x) 的概率，而不是只比较 argmax。
3. **不能忽略采样处理。** 如果 Target-only 的分布经过 temperature、top-k、top-p、repetition penalty 等处理，那么 SD 的 p 和 q 必须明确使用什么变换；residual correction 必须基于实际用于采样的 p、q。

### 2.3 Greedy 是一个特殊分支

当 `greedy=True` 时，本讲不把 greedy 伪装成普通的随机拒绝采样，而是使用确定性规则：

- Draft 提出自己的 argmax token。
- Target 对该位置的 logits 求 argmax。
- 两者相同就接受 Draft token；不同就拒绝并直接输出 Target argmax。
- 如果所有候选都被接受，再输出 Target 最后一个验证位置的 argmax bonus token。

在数学语义上，这会得到 Target-only greedy 的 token 序列。不过在真实 GPU 实现中，Target-only 的单 token decode 和 SD 的多 token verify 可能走不同 shape/kernel 路径；BF16/FP16 舍入、归约顺序和近似并列 logits 都可能造成 argmax 边界差异。因此工程上应说“算法语义上等价”，不能承诺任何硬件/任何输入都逐 token bitwise-equivalent。

### 2.4 本仓库的分布具体是什么

原 tinyInfer `Sampler` 对非 greedy 行做 temperature scaling，并保留 top-k=100 的 logits，再通过 Gumbel-Max 采样。SD helper 用 softmax 显式构造同一分布，以便计算 p(x)、q(x) 和 residual。两种采样方式在数学上对应同一 categorical distribution，但随机数消耗顺序不同，所以不能期待固定 seed 下 SD 和 Target-only 逐 token 输出完全相同。

本讲暂不实现 top-p。如果你后续加 top-p，必须对 Target logits 和 Draft logits 分别按各自排序应用 top-p 截断，再用截断后的 p、q 进行接受概率和 residual 修正。不要把“两个模型使用相同 temperature”误认为“两个模型采样分布就相同”。

## 3. Step 3：先建立缓存状态不变量，再写代码

这是本讲最重要的工程部分。定义：

- `seq.token_ids`：完整的已输出 token 历史，包括 prompt 和 completion。
- `seq.num_computed_tokens`：Target 与 Draft 共同确认、已经提交的 KV 前缀长度；Scheduler 只依据它推进下一轮。
- `seq.target_kv_computed_tokens` / `seq.draft_kv_computed_tokens`：分别记录 Target / Draft 物理 KV 中连续有效前缀的 exclusive end。SD 一轮执行中两者可以暂时不相等；只有 postprocess 完成回滚后才会重新对齐。当前实现会把每个 proposal（包括最后一个）都送入 Draft，因此 Draft proposal 结束时可能已经写到 `start + K + 1`，而 Target 仍停在 `start`；Target verify 后再追上。
- `start = seq.num_computed_tokens`：当前尚未写入已提交 KV 前缀的 pending token 所在位置。
- `K`：本轮实际生成的 Draft proposal 数，可能小于 UI 设置的上限，例如接近 `max_model_len` 或 KV block 不足时。
- `verified_end = start + K + 1`：Target 本轮验证输入 `[pending, d1, ..., dK]` 的结束位置（exclusive）。
- `A`：本轮接受的 Draft proposal 数。

在进入 SD decode 之前，正常的 prefill 已经把 prompt 的 KV 写入两份 cache，并从 Target 的最后一个 prompt token 采样出第一个输出 token。于是通常满足：

$$
\text{len(seq.token\_ids)}=\text{seq.num\_computed\_tokens}+1
$$

也就是说，最后一个输出 token 已经存在于 token 历史中，但它的 KV 还没有写入 cache；它就是本轮的 pending token。

Target 验证的输入不是单独的 Draft 候选，而是：

```text
[pending, d1, d2, ..., dK]
```

Target forward 一次处理这 K+1 个输入位置：第 0 行 logits 用来验证 d1，第 1 行 logits 用来验证 d2，依此类推；最后第 K 行 logits 在所有 proposal 被接受时用于生成 bonus token。

### 3.1 这一轮结束时，KV 应该保留到哪里

如果接受了 A 个 Draft proposal，那么 pending token 一定是已确认的输入，因为它在本轮开始前已经由 Target 生成；它后面的 A 个 accepted proposal 也可以提交。因此逻辑 KV 前缀长度为：

$$
\text{keep\_tokens}=\text{start}+1+A
$$

如果发生拒绝，correction token 是新输出，但 Target 没有把 correction token 作为输入计算过 KV；它应该留在 `seq.token_ids` 中，等待下一轮成为 pending token。所有被拒绝的 proposal 以及其后候选都不能被后续 attention 读取。

如果 K 个 proposal 全部接受，Target cache 正好提交到 `verified_end`，而 bonus token 尚未进入 KV。下一轮会把 bonus token 作为 pending 输入计算。

### 3.2 用具体例子看拒绝回滚

假设本轮开始时 `start=100`，Draft 提出 4 个候选：`d1,d2,d3,d4`。Target 验证输入是位置 100 到 104，共 5 个 token。前两个候选被接受，第三个被拒绝，因此 `A=2`：

```text
位置             100       101       102       103       104
Target 输入       pending   d1        d2        d3        d4
Target KV         已写入    已写入    已写入    已写入    已写入
SD 决策           固定前缀  accept    accept    reject    discard
最终逻辑前缀      [位置 0 ... 102]，长度 keep_tokens = 103
```

Target forward 确实已经物理写入位置 103、104 的 KV，但这两处数据不再属于已确认前缀。下一步需要：

1. 把 correction token 追加到 `seq.token_ids`；
2. 把 `seq.num_computed_tokens` 设为 103，而不是 105；
3. 分别按 `target_kv_computed_tokens` 和 `draft_kv_computed_tokens` 清理各自写过的无效尾部；然后把两份 frontier 都回退到 103。清零是教学实现的防御性措施；
4. 保留 `seq.block_table` 不变。即使本轮预留了位置 104 所在的新 block，回滚也不释放、不删除这个 block ID；它仍属于当前请求的 reservation，后续真正写入时继续复用；
5. 只对 `min(target_frontier, draft_frontier, upto_token)` 覆盖到的完整 block 注册 Prefix Cache hash；注册和请求结束时的释放仍然由 Scheduler 的 postprocess 触发；
6. 下一轮从位置 103 读取 correction token，分别更新 Target 和 Draft KV，并开始新一轮 proposal。

**回滚不是删除已输出的 accepted token，也不是简单地把 `seq.token_ids` 截断到 keep_tokens。** correction token 必须保留在序列中，但其 KV 尚未计算，所以它会位于 `num_computed_tokens` 指向的位置。

### 3.3 为什么 Target 和 Draft 必须各有一份 cache

Target 与 Draft 的层数、hidden size、head 数可能不同，K/V 张量的 shape 也不同。不能让两模型共用同一个 `PagedKVCache.storage`。本讲创建两份独立 cache：

```text
BlockManager logical block IDs: [0, 1, 2, ...]
                 │
                 ├── target_kv_cache.storage[... block_id ...]
                 └── draft_kv_cache.storage[... block_id ...]
```

两份 cache 使用相同的 `block_id` 和 `block_table` 映射，是为了让现有 BlockManager 的请求所有权、Prefix Cache 和 UI 方格继续只维护一份逻辑 block metadata。相同 block ID 在两份 storage 中占用的字节数不必相同；Target block bytes 和 Draft block bytes 分别计算。一个逻辑 block 的 hash 仍然只由 parent hash 与完整 token IDs 计算一次；是否允许注册，则由双模型的有效 KV frontier 决定。

### 3.4 为什么统一 Prefix Cache 命中后不需要给 Draft 补算前缀

本讲采用统一管理方案：一个逻辑 Block 对应一段 token IDs、一份 hash / parent hash / token metadata，以及两份独立的 KV storage slot——`target_kv_cache.storage[..., block_id, ...]` 和 `draft_kv_cache.storage[..., block_id, ...]`。hash 只由 token IDs 和 parent hash 计算一次，因为两模型处理的是同一段 token 历史；K/V 数值仍然分别由各自模型计算，绝不共享 tensor 内容。

关键约束是：**只有 Target 和 Draft 都已将某逻辑 block 的全部 `block_size` 个 token 写入各自 KV，`cache_computed_full_blocks()` 才能为该 block 绑定 hash 并加入 `hash_to_block_ids`。** 因此，Prefix Cache 中任何可命中的 block 都已经是 paired-complete block。Admission 命中后，把同一个 logical block ID 加入 `seq.block_table`，就同时复用了 Target 和 Draft 的对应 KV；不需要 `_prime_draft_prefixes()`，也不需要 Draft 再对命中前缀 forward。

这不是“同一个 hash 自动证明两份 storage 有效”，而是“只有双 KV 都完整有效时才允许把该 hash 注册进去”。新分配或被 eviction 的 block 在重新写满之前没有可命中的 hash。该保证以同一个引擎实例内模型权重、tokenizer、RoPE / cache dtype 等配置固定为前提；更换模型或相关配置时，应清空该引擎的 Prefix Cache。

## 4. Step 4：双模型显存账本和 block 数估算

对某模型，单个 KV block 的理论字节数为：

$$
B_{\text{block}}=L\times 2\times S\times H_{kv}\times D\times b
$$

其中 L 为层数，2 代表 K 和 V，S 为 block size，Hkv 为 KV head 数，D 为 head dimension，b 为每个元素的字节数。Target 和 Draft 分别计算：

$$
B_t=L_t\times 2\times S\times H_{kv,t}\times D_t\times b
$$

$$
B_d=L_d\times 2\times S\times H_{kv,d}\times D_d\times b
$$

因为一个逻辑 block ID 会在两份 storage 中各占一个 block，分配一个逻辑 block 的总成本为：

$$
B_{\text{paired}}=B_t+B_d
$$

本讲的 profile 顺序为：先加载 Target 与 Draft 权重；分别对 Target、Draft 做一次 dummy prefill profile；每次 profile 时都保留两模型权重常驻，但只临时创建当前被 profile 模型的 dummy KV；用峰值减去两模型权重和当前 dummy KV，估算两种 forward 中更大的临时峰值。最终 block 数为：

$$
N_{\text{blocks}}=\left\lfloor\frac{M_{\text{budget}}-W_t-W_d-T_{\text{peak}}}{B_t+B_d}\right\rfloor
$$

$$
M_{\text{budget}}=M_{\text{total}}\times \text{gpu\_memory\_utilization}
$$

该 profile 是面向本项目的保守预算估算，不是严格的生产级显存保证：`torch.cuda.max_memory_allocated()` 统计 PyTorch allocator 的已分配峰值，未必覆盖所有外部 CUDA workspace；不同输入 shape、attention backend、allocator 碎片和其他 GPU 进程也会影响真实峰值。

### 4.1 UI 的资源数值如何理解

- `Weights` 统计 Target + Draft 两份参数。
- `Temporary Peak` 采用两个模型 dummy prefill 的非权重/非 dummy-KV 峰值估计中的较大值。
- `Allocated For KV` 按 active logical block 数乘以 `B_paired` 计算，所以在 SD 模式下每个 active block 同时计入 Target 与 Draft 两份 cache。
- `Available For KV` 是同一预算口径下剩余的逻辑预算。
- `KV cache blocks` 的方格只画 Target 逻辑 block ID 状态；一个方格对应同 ID 的 Target block 和 Draft block。方格旁显示两者每 block 的字节数以及两份 cache 的预分配 pool 大小。

重要：`PagedKVCache.storage` 在初始化时整池 `torch.empty` 预分配，因此 UI 的 Allocated/Available 仍然是**逻辑预算分布**，并不等同于每一时刻的 `torch.cuda.memory_allocated()` 物理分配变化。该口径沿用 advance06 的设计；本讲扩展的是双模型/双 cache 的成本计算，而不是把已有图表误称为 allocator 的实时物理计数。

## 5. Step 5：文件修改清单与实施顺序

从仓库根目录 `tinyInfer-master/` 执行后续命令。按下表顺序操作，每个修改文件的完整内容都在本讲后续章节中给出。

| 顺序 | 文件 | 修改目的 |
|---|---|---|
| 1 | `tinyinfer/config.py` | 增加 SD 开关、Draft 路径、draft 长度和 adaptive 配置；基线默认关闭 |
| 2 | `tinyinfer/engine/sequence.py` | 显式记录 Target / Draft 各自的物理 KV frontier，同时保留共同提交 frontier |
| 3 | `tinyinfer/engine/block_manager.py` | 统一管理逻辑 block 与单一 token hash；只有双模型完整 block 才注册 Prefix Cache；reservation 在 rollback 后保留 |
| 4 | `tinyinfer/engine/speculative.py` | 新增可单测的分布构造、拒绝采样和按模型各自 frontier 清理 cache tail 的函数 |
| 5 | `tinyinfer/engine/model_runner.py` | 加载两模型、双 KV cache 和双模型显存 profile；并行 prefill；批量 Draft proposal / Target verify；记录双 frontier 与 cache rollback |
| 6 | `tinyinfer/engine/scheduler.py` | 在 postprocess 中提交共同前缀、注册双 KV 完整 block、处理 correction/bonus/EOS，并在结束时释放 reservation |
| 7 | `tinyinfer/engine/llm_engine.py` | 用配置开关分流原始路径与 SD 路径，保留现有调度和流式更新 |
| 8 | `tinyinfer/runtime/dynamic_engine.py` | 把 SD 指标与运行配置随 runtime snapshot 推送；提供动态配置入口 |
| 9 | `tinyinfer/ui/app.py` | 增加 WebSocket 的 `configure_sd` 控制消息 |
| 10 | `tinyinfer/ui/static/index.html` | 保留原 UI，添加 SD 指标和全局 draft 控制 |
| 11 | `tinyinfer/ui/static/app.js` | 绘制双模型资源口径、SD 统计和动态控制 |
| 12 | `tinyinfer/ui/static/style.css` | SD 控制布局与样式 |
| 13 | `examples/advance07_speculative_decoding.py` | 新增 CLI，同时支持 baseline 和 SD 模式 |
| 14 | `tests/test_speculative_decoding.py` | 测试概率分布、接受/拒绝修正、双 frontier 回滚、paired Prefix Cache 与 reservation 保留 |

旧文件中的新增/修改行会在完整代码中用 `# [MOD]` 或对应语言的注释标记。新文件整份都是新增内容，因此不在每一行重复插入注释，以免影响阅读。

## 6. Step 6：修改 `tinyinfer/config.py`

SD 默认关闭，原本不传任何 SD 参数时仍然使用 Target-only 路径。`draft_hf_config` 仅在 `speculative=True` 时加载；`max_model_len` 同时受 Target 与 Draft 的最大位置限制。

完整文件：

**完整文件：** `tinyinfer/config.py`

```python
from transformers import AutoConfig
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str | None = None                # 理论模型地址, 如果是本地文件需要写完整路径
    max_num_batched_tokens: int = 4096      # 一轮调度中, 所有 Sequence 加起来, 最多允许送进模型计算多少个 token
    max_num_seqs: int = 64                  # 一轮调度中, 最多同时允许多少条请求同时推理 
    max_model_len: int = 4096               # 单条 Sequence 最多允许多长的总上下文; 实际单条sequence上下文长度上限 = min(max_model_len, prompt + sampling_params.max_tokens)
    gpu_memory_utilization: float = 0.90    # 允许 tinyInfer 最多使用 GPU 显存的大约比例(用于KV Cache blocks计算)
    tensor_parallel_size: int = 1           # 单卡默认TP=1, 后期会在单卡上模拟TP = 2/4
    enforce_eager: bool = True              # 是否强制使用普通 eager execution; 如果为False可以启用CUDA Graph
    kvcache_block_size: int = 256           # 一个 KV Cache block 容纳多少个 token 的 K/V

    # Speculative Decoding is opt-in; baseline behavior stays unchanged. # [MOD]
    speculative: bool = False # [MOD]
    draft_model: str | None = None # [MOD]
    max_draft_tokens: int = 5 # [MOD]
    adaptive_draft: bool = False # [MOD]

    hf_config: Any | None = None
    draft_hf_config: Any | None = None # [MOD]
    eos_token_id: int = -1
    num_kvcache_blocks: int = 0             # 运行时根据 GPU 剩余显存计算出来的KV Cache block总数量

    def load_hf_config(self):
        if self.model is None:
            raise ValueError("model path/name is required for real runtime")
        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        if self.max_model_len > self.hf_config.max_position_embeddings:
            self.max_model_len = self.hf_config.max_position_embeddings

        if self.speculative: # [MOD]
            if not self.draft_model: # [MOD]
                raise ValueError("--speculative requires a draft_model path") # [MOD]
            self.draft_hf_config = AutoConfig.from_pretrained( # [MOD]
                self.draft_model, # [MOD]
                trust_remote_code=True, # [MOD]
            ) # [MOD]
            if self.max_model_len > self.draft_hf_config.max_position_embeddings: # [MOD]
                self.max_model_len = self.draft_hf_config.max_position_embeddings # [MOD]




    def validate_runtime_fields(self):
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if not 1 <= self.max_draft_tokens <= 32: # [MOD]
            raise ValueError("max_draft_tokens must be in [1, 32]") # [MOD]
        if self.speculative and not self.draft_model: # [MOD]
            raise ValueError("speculative mode requires draft_model") # [MOD]
```

## Step 6A：修改 `tinyinfer/engine/sequence.py`，区分 committed frontier 与两份物理 KV frontier

`seq.num_computed_tokens` 继续承担原框架的调度语义：它只表示 Target 与 Draft 都确认有效的共同前缀。另加两个字段跟踪模型各自真实写入到哪里。这样 Draft 在 proposal 阶段先向前写、Target 仍停留在 `start` 时，不会误把尚未验证的 Draft KV 当成已提交状态。

**完整文件：** `tinyinfer/engine/sequence.py`

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
        self.num_computed_tokens = 0            # [MOD] Target 与 Draft 共同确认的 committed KV frontier；Scheduler 只推进这个值
        self.target_kv_computed_tokens = 0       # [MOD] Target storage 中连续有效 KV 的 exclusive end；SD 轮内可能暂时领先/落后
        self.draft_kv_computed_tokens = 0        # [MOD] Draft storage 中连续有效 KV 的 exclusive end；SD 轮内可能暂时领先/落后
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
            "cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,

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

## Step 6B：修改 `tinyinfer/engine/block_manager.py`，统一管理 paired KV block 与 Prefix Cache

本步骤保留一个 `Block`、一套 `hash_to_block_ids` 和一个 `block_table`。同一个 physical/logical block ID 分别索引 Target 与 Draft 的独立 KV tensor storage。Admission 命中的是已经通过双 frontier 检查后注册的 block，因此无需在命中后再给 Draft 重建前缀。

`cache_computed_full_blocks(..., require_draft=True)` 仅在 `min(upto_token, target_kv_computed_tokens, draft_kv_computed_tokens)` 覆盖到完整 block 时注册 hash。`ensure_capacity_for_token_position()` 只负责预留 block；rollback 不会回收 reservation。Prefix Cache 的注册、以及请求完成后的 `free(seq)`，仍由 Scheduler 的 postprocess 路径调用。

**完整文件：** `tinyinfer/engine/block_manager.py`

```python
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
            block = self._allocate_block()  # [MOD] 统一使用 reservation-aware allocator
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
        seq.target_kv_computed_tokens = cached_tokens  # [MOD] hit block 只可能来自双 KV 完整提交的 Prefix Cache
        seq.draft_kv_computed_tokens = cached_tokens  # [MOD] 同一逻辑 block ID 同时命中 Draft storage
        seq.last_block_hash = prev_hash
        return True

    # 在一轮推理结束后调用, 无论是decode还是chunked prefill
    def cache_computed_full_blocks(
        self, seq: Sequence, upto_token: int, require_draft: bool = False
    ):
        # Prefix registration is called from Scheduler.postprocess only. In SD mode,
        # a block is cacheable only if both model storages have completed the full block. # [MOD]
        cache_upto = int(upto_token)
        if require_draft:  # [MOD]
            cache_upto = min(
                cache_upto,
                int(seq.target_kv_computed_tokens),
                int(seq.draft_kv_computed_tokens),
            )
        full_blocks = cache_upto // self.block_size  # [MOD]

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

            # 当前物理块理论上在这一刻第一次“内容稳定”; 如果此前有其他 cache identity，先清旧索引
            self._remove_cache_index(block)
            block.bind(token_ids, block_hash, prev_hash) # 正式绑定
            self.hash_to_block_ids.setdefault(block_hash, set()).add(block_id) # 原本仅删除旧记录, 此时写入新记录
            prev_hash = block_hash

        seq.last_block_hash = prev_hash

    # 预留共享 logical block_table 的容量；token_position 是任一模型可能写入的最高逻辑位置。 # [MOD]
    # reservation 不等于 cache commit：rollback 不缩短 block_table、不释放 block ID。 # [MOD]
    # 只有请求结束时由 postprocess 调用 free(seq) 才会释放所有 active reservations。 # [MOD]
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
        block = self._allocate_block()  # [MOD] 与其他路径共用同一个 allocator
        seq.block_table.append(block.block_id)
        return True
```

## Step 7：新增拒绝采样与回滚 helper

新增 `tinyinfer/engine/speculative.py`，把 SD 的概率逻辑从模型执行流程中分离出来，方便 CPU 单元测试。

- `distribution_from_logits()`：实现与当前 Sampler 相同的 temperature + top-k=100 概率分布。
- `speculative_accept_or_correct()`：先按 `min(1, p(x)/q(x))` 接受候选，否则按 `max(p-q, 0)` 采样 correction。
- `clear_kv_cache_tail()`：按传入模型自己的 `written_end` 清理该模型在逻辑有效前缀之后实际写过的无效尾部。Target 与 Draft 的写入 frontier 可以不同，因此分别调用该 helper，而不是假设两份 cache 的长度始终相等。清零是防御性措施；正确性首先来自共同 committed frontier 与 attention context 长度。

**完整文件：** `tinyinfer/engine/speculative.py`

```python
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SpeculativeResult:
    token_ids: list[int]
    accepted_count: int
    verified_end: int  # Target verify 写入位置的 exclusive end
    draft_written_end: int  # Draft proposal 阶段实际写入位置的 exclusive end
    proposed_count: int


def distribution_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int = 100,
) -> torch.Tensor:
    """Return the sampling distribution used by tinyInfer's non-greedy sampler."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scaled = logits.float() / float(temperature)
    k = min(int(top_k), scaled.numel())
    threshold = torch.topk(scaled, k=k, dim=-1).values[-1]
    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    return torch.softmax(scaled, dim=-1)


def sample_distribution(probs: torch.Tensor) -> int:
    if probs.ndim != 1:
        raise ValueError("probs must be one-dimensional")
    return int(torch.multinomial(probs, num_samples=1).item())


def speculative_acceptance_mask(
    proposals: list[int], target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> list[bool]:
    """Vectorized acceptance tests; synchronize once for all K proposals."""
    if target_probs.ndim != 2 or draft_probs.shape != target_probs.shape:
        raise ValueError("target/draft distributions must have the same [K, vocab] shape")
    if len(proposals) != target_probs.shape[0]:
        raise ValueError("proposal count must equal the distribution batch size")
    if not proposals:
        return []
    proposal_ids = torch.tensor(proposals, dtype=torch.long, device=target_probs.device)
    rows = torch.arange(len(proposals), dtype=torch.long, device=target_probs.device)
    p_x = target_probs[rows, proposal_ids]
    q_x = draft_probs[rows, proposal_ids].clamp_min(torch.finfo(draft_probs.dtype).tiny)
    accept_probs = torch.clamp(p_x / q_x, max=1.0)
    # .tolist() performs one device-to-host synchronization for the whole round.
    return (torch.rand_like(accept_probs) < accept_probs).tolist()


def sample_residual_correction(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> int:
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 1:
        raise ValueError("target/draft distributions must have the same 1-D shape")
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    mass = residual.sum()
    if float(mass.item()) <= 1e-12:
        # Mathematically unreachable after a true rejection; numerical fallback.
        residual = target_probs
    else:
        residual = residual / mass
    return int(torch.multinomial(residual, num_samples=1).item())


def speculative_accept_or_correct(
    proposal: int,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> tuple[bool, int]:
    """Exact rejection sampler for one proposal, up to floating-point error."""
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 1:
        raise ValueError("target/draft distributions must have the same 1-D shape")
    if not 0 <= proposal < target_probs.numel():
        raise ValueError("proposal token is out of range")

    q_x = float(draft_probs[proposal].item())
    if q_x <= 0.0:
        raise RuntimeError("draft proposed a token with zero draft probability")
    p_x = float(target_probs[proposal].item())
    accept_prob = min(1.0, p_x / q_x)

    if float(torch.rand((), device=target_probs.device).item()) < accept_prob:
        return True, int(proposal)

    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    mass = residual.sum()
    if float(mass.item()) <= 1e-12:
        # The rejection branch is mathematically unreachable when p == q; guard
        # against a zero residual caused by finite-precision rounding.
        residual = target_probs.clone()
        mass = residual.sum()
    residual = residual / mass
    return False, sample_distribution(residual)


def clear_kv_cache_tail(
    caches, block_table: list[int], block_size: int, keep_tokens: int, written_end: int
) -> None:
    """Zero one model's invalid speculative tail up to its own written frontier."""
    if keep_tokens < 0 or written_end < keep_tokens:
        raise ValueError("invalid speculative rollback range")
    token_position = keep_tokens
    while token_position < written_end:
        logical_block = token_position // block_size
        offset = token_position % block_size
        if logical_block >= len(block_table):
            raise RuntimeError("rollback position has no allocated physical block")
        take = min(written_end - token_position, block_size - offset)
        physical_block = block_table[logical_block]
        for cache in caches:
            cache.storage[:, :, physical_block, offset:offset + take].zero_()
        token_position += take
```

## Step 8：双模型加载、双 KV pool、proposal 与 Target verify

`model_runner.py` 是本讲核心文件，分成五个相对独立的部分。

1. `build_speculative_kv_memory_profile()`：在两模型权重均驻留的前提下，分别 profile 两种模型的 dummy prefill，并按 `target_block_bytes + draft_block_bytes` 计算共享逻辑 block 的数量。
2. `run_prefill_with_draft()`：prefill 阶段 Target 与 Draft 走相同输入、相同 ragged-batch metadata，但写入不同的 KV storage。Prefix Cache 命中的 block 已经保证两份 KV 都完整有效，因此直接复用同一组 block ID，不再重建 Draft 前缀。
3. `run_speculative()`：对所有 decode 用户先批量计算 pending token 的 Draft logits；随后逐 draft depth 对活跃用户做批量 Draft forward；最后把每个用户的 `[pending, proposals...]` 合并成 ragged batch，**一次 Target forward 验证所有用户的候选段**。
4. `_record_speculative_batch()`：只统计本批真正参与 SD decode 且 `proposed_count > 0` 的请求，把 proposal / accepted / verification-round 三个计数累计到 ModelRunner 全局状态；由此得到**从引擎启动到当前时刻**的全局 Acceptance Rate 和 Avg Acceptance Length。所有用户共享同一个全局 `max_draft_tokens`，本讲不再使用 EMA 或自适应 K。
5. `rollback_cache_tail()`：根据 scheduler 最终确定的 `keep_tokens`，分别按 Target / Draft 自己的写入 frontier 清理无效后缀；回滚只修改有效长度与数据，不缩短 `block_table`。

这里特意没有让 Target 与 Draft 共用一个 `PagedKVCache` 对象。两者只共享 block ID 的逻辑映射；storage、层数、head 维度与每 block 字节数都独立。

**完整文件：** `tinyinfer/engine/model_runner.py`

```python
from dataclasses import dataclass
from threading import Lock  # [MOD]

import torch

from tinyinfer.utils.context import set_context, reset_context
from tinyinfer.engine.sequence import SchedulerOutput
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler, TOP_K  # [MOD]
from tinyinfer.engine.speculative import (  # [MOD]
    SpeculativeResult,  # [MOD]
    distribution_from_logits,  # [MOD]
    sample_distribution,  # [MOD]
    speculative_acceptance_mask,  # [MOD]
    sample_residual_correction,  # [MOD]
    clear_kv_cache_tail,  # [MOD]
)  # [MOD]
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


# Resource Statistics 统一采用同一套预算口径，五部分严格相加等于 GPU 总显存：
# Reserved: TOTAL_MEM * (1 - gpu_memory_utilization), 完全不使用的预留区域
# Weights: 模型权重
# Temporary Peak: 一次最大token数prefill过程中temporary activation所占用的总显存(不包含KV cache)
# KV budget: 包含Allocated和Available, 分别代表KV cache已经占用和还可以占用的显存大小
# Reserved + Weights + Temporary Peak + Allocated for KV + Available for KV = TOTAL_MEM
@dataclass(frozen=True, slots=True)
class KVMemoryProfile:
    total_bytes: int
    budget_bytes: int
    reserved_bytes: int
    weights_bytes: int
    temporary_peak_bytes: int
    block_bytes: int
    num_blocks: int
    target_block_bytes: int = 0  # [MOD]
    draft_block_bytes: int = 0  # [MOD]
    target_weights_bytes: int = 0  # [MOD]
    draft_weights_bytes: int = 0  # [MOD]


# 模型已经驻留 GPU 后，再根据当前 reserved memory 估算 KV block 数
def build_kv_memory_profile(
    config,
    device: torch.device,
    dtype: torch.dtype,
    model
) -> KVMemoryProfile:
    if device.type != "cuda":
        raise ValueError("Invalid device: cpu; Required: cuda")
    from transformers import AutoTokenizer

    # 清掉 allocator 中可释放的历史缓存后再做 KV profile。
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = int(total_bytes * config.gpu_memory_utilization)
    reserved_bytes = total_bytes - budget_bytes
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) 
    block_bytes = bytes_per_kv_block(config.hf_config, config.kvcache_block_size, dtype)
    if block_bytes <= 0:
        raise RuntimeError("invalid KV bytes per block")

    # 直接把 hello 文本编码为指定长度，不经过 Engine/Scheduler
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    hello_ids = tokenizer.encode("hello", add_special_tokens=False)
    if not hello_ids:
        raise RuntimeError("hello tokenizer returned no tokens")

    # 创建dummy input和dummy pagedKVCache
    n = config.max_num_batched_tokens
    ids = (hello_ids * ((n + len(hello_ids) - 1) // len(hello_ids)))[:n]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    positions = torch.arange(n, dtype=torch.long, device=device)
    num_dummy_blocks = (n + config.kvcache_block_size - 1) // config.kvcache_block_size
    head_dim = int(getattr(config.hf_config, "head_dim", config.hf_config.hidden_size // config.hf_config.num_attention_heads))
    dummy_cache = PagedKVCache( 
        num_layers=config.hf_config.num_hidden_layers,
        num_blocks=num_dummy_blocks, 
        block_size=config.kvcache_block_size,
        num_kv_heads=config.hf_config.num_key_value_heads, 
        head_dim=head_dim, dtype=dtype, device=device, 
    ) 
    dummy_kv_bytes = dummy_cache.size()
    model.set_kv_cache(dummy_cache) 

    try: # 直接跑一次forward
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device) 
        with torch.inference_mode(): 
            set_context( 
                cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=device), 
                context_lens=torch.tensor([n], dtype=torch.int32, device=device),
                block_tables=torch.arange(num_dummy_blocks, dtype=torch.int32, device=device).unsqueeze(0), 
                slot_mapping=torch.arange(n, dtype=torch.long, device=device),
            ) 
            model(input_ids, positions, output_hidden_states=False) 
        torch.cuda.synchronize(device) 
        profile_peak_bytes = int(torch.cuda.max_memory_allocated(device)) 
    finally: 
        reset_context() 
        for layer in model.model.layers: 
            layer.self_attn.attn.kv_cache = None 
        del dummy_cache, input_ids, positions 
        torch.cuda.empty_cache() 

    
    temporary_peak_bytes = max(0, profile_peak_bytes - weights_bytes - dummy_kv_bytes) # 注意peak应该减去weights和dummy cache大小
    kv_budget_bytes = budget_bytes - weights_bytes - temporary_peak_bytes
    num_blocks = kv_budget_bytes // block_bytes
    if num_blocks <= 0:
        raise RuntimeError("no memory left for KV cache after dummy prefill profiling")
    
    return KVMemoryProfile(
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        reserved_bytes=reserved_bytes,
        weights_bytes=weights_bytes,
        temporary_peak_bytes=temporary_peak_bytes,
        block_bytes=int(block_bytes),
        num_blocks=int(num_blocks),
        target_block_bytes=int(block_bytes),  # [MOD]
        draft_block_bytes=0,  # [MOD]
        target_weights_bytes=int(weights_bytes),  # [MOD]
        draft_weights_bytes=0,  # [MOD]
    )



def _profile_one_model_peak(  # [MOD]
    model, hf_config, config, device: torch.device, dtype: torch.dtype, weights_bytes: int  # [MOD]
) -> tuple[int, int]:  # [MOD]
    """Return (activation peak, dummy KV bytes) while all model weights stay resident."""  # [MOD]
    n = min(config.max_num_batched_tokens, config.max_model_len)  # [MOD]
    if n <= 0:  # [MOD]
        raise ValueError("profile token count must be positive")  # [MOD]
    block_size = config.kvcache_block_size  # [MOD]
    num_dummy_blocks = (n + block_size - 1) // block_size  # [MOD]
    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))  # [MOD]
    dummy_cache = PagedKVCache(  # [MOD]
        num_layers=hf_config.num_hidden_layers,  # [MOD]
        num_blocks=num_dummy_blocks,  # [MOD]
        block_size=block_size,  # [MOD]
        num_kv_heads=hf_config.num_key_value_heads,  # [MOD]
        head_dim=head_dim,  # [MOD]
        dtype=dtype,  # [MOD]
        device=device,  # [MOD]
    )  # [MOD]
    model.set_kv_cache(dummy_cache)  # [MOD]
    input_ids = torch.zeros(n, dtype=torch.long, device=device)  # [MOD]
    positions = torch.arange(n, dtype=torch.long, device=device)  # [MOD]
    hidden = None  # [MOD]
    try:  # [MOD]
        torch.cuda.synchronize(device)  # [MOD]
        torch.cuda.reset_peak_memory_stats(device)  # [MOD]
        set_context(  # [MOD]
            q_lens=torch.tensor([n], dtype=torch.int32, device=device),  # [MOD]
            context_lens=torch.tensor([n], dtype=torch.int32, device=device),  # [MOD]
            cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=device),  # [MOD]
            max_seqlen_q=n,  # [MOD]
            slot_mapping=torch.arange(n, dtype=torch.long, device=device),  # [MOD]
            block_tables=torch.arange(num_dummy_blocks, dtype=torch.int32, device=device).unsqueeze(0),  # [MOD]
            is_prefill=torch.tensor([True], dtype=torch.bool, device=device),  # [MOD]
            is_sample=torch.tensor([True], dtype=torch.bool, device=device),  # [MOD]
        )  # [MOD]
        with torch.inference_mode():  # [MOD]
            hidden, _ = model(input_ids, positions, output_hidden_states=False)  # [MOD]
            # SD verification materializes logits for multiple token positions. # [MOD]
            # Include the LM-head allocation in the temporary peak estimate. # [MOD]
            model.compute_logits(hidden) # [MOD]
        torch.cuda.synchronize(device)  # [MOD]
        peak = int(torch.cuda.max_memory_allocated(device))  # [MOD]
        dummy_bytes = dummy_cache.size()  # [MOD]
    finally:  # [MOD]
        reset_context()  # [MOD]
        for layer in model.model.layers:  # [MOD]
            layer.self_attn.attn.kv_cache = None  # [MOD]
        del dummy_cache, input_ids, positions, hidden  # [MOD]
        torch.cuda.empty_cache()  # [MOD]
    return max(0, peak - weights_bytes - dummy_bytes), int(dummy_bytes)  # [MOD]


def build_speculative_kv_memory_profile(  # [MOD]
    config, device: torch.device, dtype: torch.dtype, target_model, draft_model, draft_hf_config  # [MOD]
) -> KVMemoryProfile:  # [MOD]
    if device.type != "cuda":  # [MOD]
        raise ValueError("Invalid device: cpu; Required: cuda")  # [MOD]
    torch.cuda.synchronize(device)  # [MOD]
    torch.cuda.empty_cache()  # [MOD]
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)  # [MOD]
    budget_bytes = int(total_bytes * config.gpu_memory_utilization)  # [MOD]
    reserved_bytes = total_bytes - budget_bytes  # [MOD]
    target_weights_bytes = sum(p.numel() * p.element_size() for p in target_model.parameters())  # [MOD]
    draft_weights_bytes = sum(p.numel() * p.element_size() for p in draft_model.parameters())  # [MOD]
    weights_bytes = target_weights_bytes + draft_weights_bytes  # [MOD]
    target_block_bytes = bytes_per_kv_block(config.hf_config, config.kvcache_block_size, dtype)  # [MOD]
    draft_block_bytes = bytes_per_kv_block(draft_hf_config, config.kvcache_block_size, dtype)  # [MOD]
    combined_block_bytes = target_block_bytes + draft_block_bytes  # [MOD]
    if combined_block_bytes <= 0:  # [MOD]
        raise RuntimeError("invalid combined KV bytes per logical block")  # [MOD]

    target_temp, _ = _profile_one_model_peak(  # [MOD]
        target_model, config.hf_config, config, device, dtype, weights_bytes  # [MOD]
    )  # [MOD]
    draft_temp, _ = _profile_one_model_peak(  # [MOD]
        draft_model, draft_hf_config, config, device, dtype, weights_bytes  # [MOD]
    )  # [MOD]
    temporary_peak_bytes = max(target_temp, draft_temp)  # [MOD]
    kv_budget_bytes = budget_bytes - weights_bytes - temporary_peak_bytes  # [MOD]
    num_blocks = kv_budget_bytes // combined_block_bytes  # [MOD]
    if num_blocks <= 0:  # [MOD]
        raise RuntimeError(  # [MOD]
            "no memory left for paired target/draft KV blocks; reduce "  # [MOD]
            "max_num_batched_tokens, gpu_memory_utilization, or model size"  # [MOD]
        )  # [MOD]
    return KVMemoryProfile(  # [MOD]
        total_bytes=total_bytes,  # [MOD]
        budget_bytes=budget_bytes,  # [MOD]
        reserved_bytes=reserved_bytes,  # [MOD]
        weights_bytes=int(weights_bytes),  # [MOD]
        temporary_peak_bytes=int(temporary_peak_bytes),  # [MOD]
        block_bytes=int(combined_block_bytes),  # [MOD]
        num_blocks=int(num_blocks),  # [MOD]
        target_block_bytes=int(target_block_bytes),  # [MOD]
        draft_block_bytes=int(draft_block_bytes),  # [MOD]
        target_weights_bytes=int(target_weights_bytes),  # [MOD]
        draft_weights_bytes=int(draft_weights_bytes),  # [MOD]
    )  # [MOD]



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
    def __init__(self, config, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):  # [MOD]
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.sampler = Sampler()
        self.draft_model = None  # [MOD]
        self.draft_kv_cache = None  # [MOD]
        self._sd_lock = Lock()  # [MOD]
        self._max_draft_tokens = int(config.max_draft_tokens)  # [MOD] 全局共享的最大 Draft 长度
        # 以下三个计数器从 Engine 启动后持续累计，不按 batch / user 清零。 # [MOD]
        self._drafted_tokens = 0  # [MOD] 累计 proposal token 数
        self._accepted_tokens = 0  # [MOD] 累计 accepted proposal token 数
        self._verification_rounds = 0  # [MOD] 累计有效 SD decode request-round 数

        if self.config.hf_config is None:
            self.config.load_hf_config()
        hf_config = self.config.hf_config
        hf_config.kvcache_block_size = self.config.kvcache_block_size

        self.model = Qwen3ForCausalLM(hf_config, kv_cache=None).to(  # [MOD]
            device=self.device, dtype=self.dtype  # [MOD]
        )  # [MOD]
        self.load_report = load_weights(self.model, self.config.model, strict=True)  # [MOD]
        self.model.eval()

        if self.config.speculative:  # [MOD]
            if self.config.draft_hf_config is None:  # [MOD]
                self.config.load_hf_config()  # [MOD]
            draft_hf_config = self.config.draft_hf_config  # [MOD]
            draft_hf_config.kvcache_block_size = self.config.kvcache_block_size  # [MOD]
            if int(hf_config.vocab_size) != int(draft_hf_config.vocab_size):  # [MOD]
                raise ValueError(  # [MOD]
                    "target and draft vocab_size differ; speculative decoding requires "  # [MOD]
                    "the same token ID space"  # [MOD]
                )  # [MOD]
            if getattr(hf_config, "model_type", None) != getattr(draft_hf_config, "model_type", None):  # [MOD]
                raise ValueError("target and draft model_type must match in this Qwen3 runtime")  # [MOD]

            self.draft_model = Qwen3ForCausalLM(draft_hf_config, kv_cache=None).to(  # [MOD]
                device=self.device, dtype=self.dtype  # [MOD]
            )  # [MOD]
            self.draft_load_report = load_weights(  # [MOD]
                self.draft_model, self.config.draft_model, strict=True  # [MOD]
            )  # [MOD]
            self.draft_model.eval()  # [MOD]
            self.resource_profile = build_speculative_kv_memory_profile(  # [MOD]
                self.config,  # [MOD]
                self.device,  # [MOD]
                self.dtype,  # [MOD]
                self.model,  # [MOD]
                self.draft_model,  # [MOD]
                draft_hf_config,  # [MOD]
            )  # [MOD]
        else:  # [MOD]
            self.resource_profile = build_kv_memory_profile(  # [MOD]
                self.config, self.device, self.dtype, self.model  # [MOD]
            )  # [MOD]

        num_blocks = self.resource_profile.num_blocks
        self.config.num_kvcache_blocks = num_blocks
        head_dim = int(
            getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
        )
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.config.kvcache_block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.model.set_kv_cache(self.kv_cache)

        if self.config.speculative:  # [MOD]
            draft_hf_config = self.config.draft_hf_config  # [MOD]
            draft_head_dim = int(  # [MOD]
                getattr(  # [MOD]
                    draft_hf_config,  # [MOD]
                    "head_dim",  # [MOD]
                    draft_hf_config.hidden_size // draft_hf_config.num_attention_heads,  # [MOD]
                )  # [MOD]
            )  # [MOD]
            self.draft_kv_cache = PagedKVCache(  # [MOD]
                num_layers=draft_hf_config.num_hidden_layers,  # [MOD]
                num_blocks=num_blocks,  # [MOD]
                block_size=self.config.kvcache_block_size,  # [MOD]
                num_kv_heads=draft_hf_config.num_key_value_heads,  # [MOD]
                head_dim=draft_head_dim,  # [MOD]
                dtype=self.dtype,  # [MOD]
                device=self.device,  # [MOD]
            )  # [MOD]
            self.draft_model.set_kv_cache(self.draft_kv_cache)  # [MOD]

    @property  # [MOD]
    def speculative_enabled(self) -> bool:  # [MOD]
        return bool(self.config.speculative)  # [MOD]

    def set_max_draft_tokens(self, value: int) -> None:  # [MOD]
        value = int(value)  # [MOD]
        if not 2 <= value <= 10:  # [MOD] 与 UI slider 的允许范围保持一致
            raise ValueError("max_draft_tokens must be in [2, 10]")  # [MOD]
        with self._sd_lock:  # [MOD]
            self._max_draft_tokens = value  # [MOD]
            self.config.max_draft_tokens = value  # [MOD]

    def speculative_snapshot(self) -> dict:  # [MOD]
        """Return global SD statistics accumulated since engine startup."""  # [MOD]
        with self._sd_lock:  # [MOD]
            proposed = self._drafted_tokens  # [MOD]
            accepted = self._accepted_tokens  # [MOD]
            rounds = self._verification_rounds  # [MOD]
            return {  # [MOD]
                "enabled": self.speculative_enabled,  # [MOD]
                "max_draft_tokens": self._max_draft_tokens,  # [MOD]
                "drafted_tokens": proposed,  # [MOD] 保留原始累计量便于调试
                "accepted_tokens": accepted,  # [MOD]
                "verification_rounds": rounds,  # [MOD]
                "acceptance_rate": accepted / proposed if proposed else None,  # [MOD]
                "avg_acceptance_length": accepted / rounds if rounds else None,  # [MOD]
            }  # [MOD]

    def _draft_limit(self, remaining_output_tokens: int) -> int:  # [MOD]
        with self._sd_lock:  # [MOD]
            limit = self._max_draft_tokens  # [MOD] 所有用户共享同一个全局 K 上限
        return max(0, min(limit, remaining_output_tokens - 1))  # [MOD]

    def _record_speculative_batch(self, results: dict[int, SpeculativeResult]) -> None:  # [MOD]
        """Accumulate decode-only SD statistics once after a whole batch is verified."""  # [MOD]
        batch_proposed = 0  # [MOD]
        batch_accepted = 0  # [MOD]
        batch_rounds = 0  # [MOD]

        for result in results.values():  # [MOD]
            proposed = int(result.proposed_count)  # [MOD]
            accepted = int(result.accepted_count)  # [MOD]
            if proposed < 0 or accepted < 0 or accepted > proposed:  # [MOD]
                raise ValueError("invalid speculative statistics")  # [MOD]
            if proposed == 0:  # [MOD] 没有 proposal 的普通单-token decode 不计入 SD 接受统计
                continue  # [MOD]
            batch_proposed += proposed  # [MOD]
            batch_accepted += accepted  # [MOD]
            batch_rounds += 1  # [MOD] 每个 decode 请求完成一次 verify 记为一个 round

        with self._sd_lock:  # [MOD] 只在最终更新共享累计量时持锁
            self._drafted_tokens += batch_proposed  # [MOD]
            self._accepted_tokens += batch_accepted  # [MOD]
            self._verification_rounds += batch_rounds  # [MOD]

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



    @torch.no_grad()  # [MOD]
    def run(self, output: SchedulerOutput) -> dict[int, int]:  # [MOD]
        if self.config.speculative:  # [MOD]
            return self.run_prefill_with_draft(output)  # [MOD]
        return self._run_target_batch(output)  # [MOD]

    @torch.no_grad()  # [MOD]
    def _run_target_batch(self, output: SchedulerOutput) -> dict[int, int]:  # [MOD]
        try:  # [MOD]
            input_ids, positions = self.prepare_batch(output)  # [MOD]
            hidden_states, _ = self.model(  # [MOD]
                input_ids, positions, output_hidden_states=False  # [MOD]
            )  # [MOD]
            for item in output.items:  # [MOD]
                item.seq.target_kv_computed_tokens = item.start_pos + item.num_tokens  # [MOD]
            return self._sample_batch_output(output, hidden_states)  # [MOD]
        finally:
            reset_context()

    @torch.no_grad()  # [MOD]
    def run_prefill_with_draft(self, output: SchedulerOutput) -> dict[int, int]:  # [MOD]
        if not output.items:  # [MOD]
            return {}  # [MOD]
        if any(not item.is_prefill for item in output.items):  # [MOD]
            raise ValueError("run_prefill_with_draft accepts prefill items only")  # [MOD]
        # Prefix Cache hits already reference paired-complete Target + Draft KV blocks. # [MOD]
        for item in output.items:  # [MOD]
            seq = item.seq  # [MOD]
            if (  # [MOD]
                seq.target_kv_computed_tokens != item.start_pos  # [MOD]
                or seq.draft_kv_computed_tokens != item.start_pos  # [MOD]
            ):  # [MOD]
                raise RuntimeError(  # [MOD]
                    "prefill must start at the common Target/Draft KV frontier"  # [MOD]
                )  # [MOD]
        try:  # [MOD]
            input_ids, positions = self.prepare_batch(output)  # [MOD]
            target_hidden, _ = self.model(  # [MOD]
                input_ids, positions, output_hidden_states=False  # [MOD]
            )  # [MOD]
            for item in output.items:  # [MOD]
                item.seq.target_kv_computed_tokens = item.start_pos + item.num_tokens  # [MOD]
            # Reuse the exact same ragged-batch metadata; only the model/KV pool differs.  # [MOD]
            self.draft_model(input_ids, positions, output_hidden_states=False)  # [MOD]
            for item in output.items:  # [MOD]
                item.seq.draft_kv_computed_tokens = item.start_pos + item.num_tokens  # [MOD]
            return self._sample_batch_output(output, target_hidden)  # [MOD]
        finally:  # [MOD]
            reset_context()  # [MOD]

    def _sample_batch_output(  # [MOD]
        self, output: SchedulerOutput, hidden_states: torch.Tensor  # [MOD]
    ) -> dict[int, int]:  # [MOD]
        sample_flat_indices: list[int] = []  # [MOD]
        sample_items = []  # [MOD]
        cursor = 0  # [MOD]
        for item in output.items:  # [MOD]
            cursor += item.num_tokens  # [MOD]
            if item.sample_after:  # [MOD]
                sample_flat_indices.append(cursor - 1)  # [MOD]
                sample_items.append(item)  # [MOD]
        if not sample_items:  # [MOD]
            return {}  # [MOD]
        idx = torch.tensor(  # [MOD]
            sample_flat_indices, dtype=torch.long, device=hidden_states.device  # [MOD]
        )  # [MOD]
        sample_hidden = hidden_states.index_select(0, idx)  # [MOD]
        logits = self.model.compute_logits(sample_hidden)  # [MOD]
        temperatures = torch.tensor(  # [MOD]
            [item.seq.sampling_params.temperature for item in sample_items],  # [MOD]
            dtype=logits.dtype,  # [MOD]
            device=logits.device,  # [MOD]
        )  # [MOD]
        greedy_mask = torch.tensor(  # [MOD]
            [item.seq.sampling_params.greedy for item in sample_items],  # [MOD]
            dtype=torch.bool,  # [MOD]
            device=logits.device,  # [MOD]
        )  # [MOD]
        tokens = self.sampler(logits, temperatures, greedy_mask)  # [MOD]
        return {  # [MOD]
            item.seq.seq_id: int(token)  # [MOD]
            for item, token in zip(sample_items, tokens.tolist())  # [MOD]
        }  # [MOD]

    @torch.no_grad()  # [MOD]
    def _forward_batch_segments(  # [MOD]
        self,  # [MOD]
        model,  # [MOD]
        seqs,  # [MOD]
        token_segments: list[list[int]],  # [MOD]
        start_positions: list[int],  # [MOD]
        return_logits: bool = True,  # [MOD]
    ) -> list[torch.Tensor]:  # [MOD]
        if not seqs or len(seqs) != len(token_segments) or len(seqs) != len(start_positions):  # [MOD]
            raise ValueError("segments, sequences, and start positions must have equal non-zero length")  # [MOD]
        input_ids: list[int] = []  # [MOD]
        positions: list[int] = []  # [MOD]
        slot_mapping: list[int] = []  # [MOD]
        q_lens: list[int] = []  # [MOD]
        context_lens: list[int] = []  # [MOD]
        block_tables: list[list[int]] = []  # [MOD]
        max_blocks = max(len(seq.block_table) for seq in seqs)  # [MOD]
        for seq, tokens, start in zip(seqs, token_segments, start_positions):  # [MOD]
            if not tokens:  # [MOD]
                raise ValueError("empty segment is not supported")  # [MOD]
            pos = list(range(start, start + len(tokens)))  # [MOD]
            input_ids.extend(tokens)  # [MOD]
            positions.extend(pos)  # [MOD]
            slot_mapping.extend(token_to_slot(seq, p) for p in pos)  # [MOD]
            q_lens.append(len(tokens))  # [MOD]
            context_lens.append(start + len(tokens))  # [MOD]
            table = list(seq.block_table)  # [MOD]
            table.extend([-1] * (max_blocks - len(table)))  # [MOD]
            block_tables.append(table)  # [MOD]
        cu_q = [0]  # [MOD]
        for q_len in q_lens:  # [MOD]
            cu_q.append(cu_q[-1] + q_len)  # [MOD]
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)  # [MOD]
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)  # [MOD]
        set_context(  # [MOD]
            q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),  # [MOD]
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),  # [MOD]
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=self.device),  # [MOD]
            max_seqlen_q=max(q_lens),  # [MOD]
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),  # [MOD]
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),  # [MOD]
            is_prefill=torch.tensor([n > 1 for n in q_lens], dtype=torch.bool, device=self.device),  # [MOD]
            is_sample=torch.ones(len(seqs), dtype=torch.bool, device=self.device),  # [MOD]
        )  # [MOD]
        try:  # [MOD]
            hidden, _ = model(ids, pos_tensor, output_hidden_states=False)  # [MOD]
            frontier_attr = (  # [MOD]
                "draft_kv_computed_tokens" if model is self.draft_model  # [MOD]
                else "target_kv_computed_tokens"  # [MOD]
            )  # [MOD]
            for seq, tokens, start in zip(seqs, token_segments, start_positions):  # [MOD]
                current_frontier = int(getattr(seq, frontier_attr))  # [MOD]
                if start != current_frontier:  # [MOD]
                    raise RuntimeError(  # [MOD]
                        f"{frontier_attr} mismatch: segment starts at {start}, "  # [MOD]
                        f"but current frontier is {current_frontier}"  # [MOD]
                    )  # [MOD]
                setattr(seq, frontier_attr, start + len(tokens))  # [MOD]
            output = model.compute_logits(hidden) if return_logits else hidden  # [MOD]
            result = []  # [MOD]
            for i, q_len in enumerate(q_lens):  # [MOD]
                result.append(output[cu_q[i]:cu_q[i + 1]])  # [MOD]
            return result  # [MOD]
        finally:  # [MOD]
            reset_context()  # [MOD]

    @torch.no_grad()  # [MOD]
    def run_speculative(  # [MOD]
        self, output: SchedulerOutput, block_manager  # [MOD]
    ) -> dict[int, SpeculativeResult]:  # [MOD]
        if self.draft_model is None:  # [MOD]
            raise RuntimeError("speculative mode is enabled without a draft model")  # [MOD]
        if not output.items:  # [MOD]
            return {}  # [MOD]
        items = output.items  # [MOD]
        seqs = [item.seq for item in items]  # [MOD]
        starts = [item.start_pos for item in items]  # [MOD]
        ks: list[int] = []  # [MOD]
        pending_tokens: list[int] = []  # [MOD]
        for item in items:  # [MOD]
            if item.is_prefill or item.num_tokens != 1:  # [MOD]
                raise ValueError("speculative decode accepts one-token decode items only")  # [MOD]
            seq = item.seq  # [MOD]
            start = item.start_pos  # [MOD]
            if seq.num_computed_tokens != start or seq.num_tokens != start + 1:  # [MOD]
                raise RuntimeError(  # [MOD]
                    "SD invariant violated: decode must have exactly one uncomputed pending token"  # [MOD]
                )  # [MOD]
            if (  # [MOD]
                seq.target_kv_computed_tokens != start  # [MOD]
                or seq.draft_kv_computed_tokens != start  # [MOD]
            ):  # [MOD]
                raise RuntimeError(  # [MOD]
                    "SD round must begin with aligned Target/Draft committed KV frontiers"  # [MOD]
                )  # [MOD]
            remaining = min(  # [MOD]
                seq.sampling_params.max_tokens - seq.num_completion_tokens,  # [MOD]
                self.config.max_model_len - seq.num_tokens,  # [MOD]
            )  # [MOD]
            if remaining <= 0:  # [MOD]
                raise RuntimeError("scheduler submitted a sequence with no output budget")  # [MOD]
            k = self._draft_limit(remaining)  # [MOD]
            # One shared block_table must cover every position either model may write. # [MOD]
            # Target verifies [pending, d1, ..., dK], so its last write position is start + K. # [MOD]
            # This implementation also feeds dK through Draft, making the Draft last write start + K. # [MOD]
            # If a later optimization stops Draft before feeding dK, keep the max() reservation rule. # [MOD]
            while True:  # [MOD]
                target_last_position = start + k  # [MOD]
                draft_last_position = start + k  # [MOD]
                required_last_position = max(target_last_position, draft_last_position)  # [MOD]
                if block_manager.ensure_capacity_for_token_position(seq, required_last_position):  # [MOD]
                    break  # [MOD]
                if k == 0:  # [MOD]
                    raise RuntimeError("unable to reserve KV block for pending token")  # [MOD]
                k -= 1  # [MOD]
            # The block table is a reservation ledger; postprocess rollback never shrinks it. # [MOD]
            ks.append(k)  # [MOD]
            pending_tokens.append(seq.token_ids[start])  # [MOD]

        # Step A: consume each sequence's pending token in the draft model.  # [MOD]
        draft_logits = self._forward_batch_segments(  # [MOD]
            self.draft_model, seqs, [[token] for token in pending_tokens], starts  # [MOD]
        )  # [MOD]
        next_draft_logits = [x[-1] for x in draft_logits]  # [MOD]
        proposals: list[list[int]] = [[] for _ in seqs]  # [MOD]
        draft_distributions: list[list[torch.Tensor]] = [[] for _ in seqs]  # [MOD]

        # Step B: autoregressively draft up to K tokens. Each iteration batches  # [MOD]
        # the currently active users into one draft forward.  # [MOD]
        max_k = max(ks, default=0)  # [MOD]
        for draft_index in range(max_k):  # [MOD]
            active = [i for i, k in enumerate(ks) if k > draft_index]  # [MOD]
            greedy_active = [i for i in active if seqs[i].sampling_params.greedy]  # [MOD]
            sample_active = [i for i in active if not seqs[i].sampling_params.greedy]  # [MOD]
            token_by_index: dict[int, int] = {}  # [MOD]

            # Batch sampling by mode to avoid one GPU->CPU synchronization per user.  # [MOD]
            if greedy_active:  # [MOD]
                greedy_logits = torch.stack([next_draft_logits[i] for i in greedy_active])  # [MOD]
                greedy_tokens = greedy_logits.argmax(dim=-1).tolist()  # [MOD]
                token_by_index.update(zip(greedy_active, greedy_tokens))  # [MOD]
            if sample_active:  # [MOD]
                sample_probs = [  # [MOD]
                    distribution_from_logits(  # [MOD]
                        next_draft_logits[i], seqs[i].sampling_params.temperature, top_k=TOP_K  # [MOD]
                    )  # [MOD]
                    for i in sample_active  # [MOD]
                ]  # [MOD]
                for i, probs in zip(sample_active, sample_probs):  # [MOD]
                    draft_distributions[i].append(probs)  # [MOD]
                sampled = torch.multinomial(torch.stack(sample_probs), num_samples=1).squeeze(-1).tolist()  # [MOD]
                token_by_index.update(zip(sample_active, sampled))  # [MOD]

            active_tokens: list[int] = []  # [MOD]
            active_starts: list[int] = []  # [MOD]
            active_seqs = []  # [MOD]
            for i in active:  # [MOD]
                token = int(token_by_index[i])  # [MOD]
                proposals[i].append(token)  # [MOD]
                active_tokens.append(token)  # [MOD]
                active_starts.append(starts[i] + 1 + draft_index)  # [MOD]
                active_seqs.append(seqs[i])  # [MOD]
            if active_seqs:  # [MOD]
                next_logits = self._forward_batch_segments(  # [MOD]
                    self.draft_model,  # [MOD]
                    active_seqs,  # [MOD]
                    [[token] for token in active_tokens],  # [MOD]
                    active_starts,  # [MOD]
                )  # [MOD]
                for i, logits in zip(active, next_logits):  # [MOD]
                    next_draft_logits[i] = logits[-1]  # [MOD]

        # Step C: one ragged target forward verifies [pending, proposal_1, ...]  # [MOD]
        # for every active user in a single batched model invocation.  # [MOD]
        verify_segments = [  # [MOD]
            [pending_tokens[i], *proposals[i]] for i in range(len(seqs))  # [MOD]
        ]  # [MOD]
        target_logits = self._forward_batch_segments(  # [MOD]
            self.model, seqs, verify_segments, starts  # [MOD]
        )  # [MOD]

        results: dict[int, SpeculativeResult] = {}  # [MOD]
        for i, seq in enumerate(seqs):  # [MOD]
            params = seq.sampling_params  # [MOD]
            k = ks[i]  # [MOD]
            logits_i = target_logits[i]  # [MOD]
            emitted: list[int] = []  # [MOD]
            accepted_count = 0  # [MOD]
            rejected = False  # [MOD]
            stopped_on_eos = False  # [MOD]
            if params.greedy and k > 0:  # [MOD]
                target_greedy_tokens = logits_i[:k].argmax(dim=-1).tolist()  # [MOD]
            else:  # [MOD]
                target_greedy_tokens = []  # [MOD]
            if not params.greedy and k > 0:  # [MOD]
                target_proposal_probs = torch.stack([  # [MOD]
                    distribution_from_logits(logits_i[j], params.temperature, top_k=TOP_K)  # [MOD]
                    for j in range(k)  # [MOD]
                ])  # [MOD]
                draft_proposal_probs = torch.stack(draft_distributions[i])  # [MOD]
                accept_mask = speculative_acceptance_mask(  # [MOD]
                    proposals[i], target_proposal_probs, draft_proposal_probs  # [MOD]
                )  # [MOD]
            else:  # [MOD]
                target_proposal_probs = None  # [MOD]
                draft_proposal_probs = None  # [MOD]
                accept_mask = []  # [MOD]

            for j, proposal in enumerate(proposals[i]):  # [MOD]
                if params.greedy:  # [MOD]
                    target_token = int(target_greedy_tokens[j])  # [MOD]
                    accepted = proposal == target_token  # [MOD]
                    chosen = proposal if accepted else target_token  # [MOD]
                else:  # [MOD]
                    accepted = accept_mask[j]  # [MOD]
                    if accepted:  # [MOD]
                        chosen = proposal  # [MOD]
                    else:  # [MOD]
                        chosen = sample_residual_correction(  # [MOD]
                            target_proposal_probs[j], draft_proposal_probs[j]  # [MOD]
                        )  # [MOD]
                emitted.append(int(chosen))  # [MOD]
                if accepted:  # [MOD]
                    accepted_count += 1  # [MOD]
                else:  # [MOD]
                    rejected = True  # [MOD]
                    break  # [MOD]
                if (  # [MOD]
                    not params.ignore_eos  # [MOD]
                    and self.config.eos_token_id >= 0  # [MOD]
                    and chosen == self.config.eos_token_id  # [MOD]
                ):  # [MOD]
                    stopped_on_eos = True  # [MOD]
                    break  # [MOD]

            if not rejected and not stopped_on_eos:  # [MOD]
                if params.greedy:  # [MOD]
                    bonus = int(logits_i[k].argmax().item())  # [MOD]
                else:  # [MOD]
                    bonus_probs = distribution_from_logits(  # [MOD]
                        logits_i[k], params.temperature, top_k=TOP_K  # [MOD]
                    )  # [MOD]
                    bonus = sample_distribution(bonus_probs)  # [MOD]
                emitted.append(bonus)  # [MOD]

            verified_end = starts[i] + len(verify_segments[i])  # [MOD]
            results[seq.seq_id] = SpeculativeResult(  # [MOD]
                token_ids=emitted,  # [MOD]
                accepted_count=accepted_count,  # [MOD]
                verified_end=verified_end,  # [MOD]
                draft_written_end=int(seq.draft_kv_computed_tokens),  # [MOD]
                proposed_count=k,  # [MOD]
            )  # [MOD]
        self._record_speculative_batch(results)  # [MOD] 一批 decode 请求 verify 完成后统一累计全局统计
        return results  # [MOD]

    @torch.no_grad()  # [MOD]
    def rollback_cache_tail(  # [MOD]
        self, seq, keep_tokens: int, target_written_end: int, draft_written_end: int  # [MOD]
    ) -> None:  # [MOD]
        """Rollback each model to the same committed prefix without releasing block reservations."""  # [MOD]
        if keep_tokens > min(target_written_end, draft_written_end):  # [MOD]
            raise RuntimeError("cannot commit tokens missing from either model's KV cache")  # [MOD]
        clear_kv_cache_tail(  # [MOD]
            [self.kv_cache], seq.block_table, seq.block_size, keep_tokens, target_written_end  # [MOD]
        )  # [MOD]
        if self.draft_kv_cache is not None:  # [MOD]
            clear_kv_cache_tail(  # [MOD]
                [self.draft_kv_cache], seq.block_table, seq.block_size, keep_tokens, draft_written_end  # [MOD]
            )  # [MOD]
        # Logical reservations remain in seq.block_table even when their contents are rolled back. # [MOD]
        seq.target_kv_computed_tokens = keep_tokens  # [MOD]
        seq.draft_kv_computed_tokens = keep_tokens  # [MOD]
```

## Step 9：新增 SD 专用 Scheduler postprocess

原始 `Scheduler.postprocess()` 的语义是每个 item 更新一个 scheduled input token，并追加一个 sampled output token。SD 一轮可能追加多个 token，所以不能把 SD 结果硬塞进原始的 `sampled_tokens: dict[int, int]` 路径。

新增 `postprocess_speculative()`，每个 decode item 根据 `accepted_count` 算出 `keep_tokens`，追加本轮真正输出的 token，回滚无效 cache tail，然后只按已确认前缀注册完整 Prefix Cache blocks。完成 EOS / `max_tokens` / `max_model_len` 判断后，仍使用原有 BlockManager 释放请求资源。原始 `postprocess()` 不变，baseline 路径继续使用它。

**完整文件：** `tinyinfer/engine/scheduler.py`

```python
from collections import deque
from dataclasses import dataclass

from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence, SequenceStatus, ScheduledItem, SchedulerOutput
from tinyinfer.utils.debug import trace_enabled
from tinyinfer.engine.block_manager import BlockManager


# 调度器, 本身不负责运行前向传播, 而是选择合适的sequence给ModelRunner进行前向传播
# 待改进: 加入真正的Mixed Batching; 改进调度的公平性


class Scheduler:
    def __init__(self, config: Config):
        # 一轮最多允许多少条 Sequence 同时处于 running。
        # Day01 暂时只真正使用 max_num_seqs；
        # max_num_batched_tokens 会在后续 token-budget / chunked prefill 中使用。
        self.max_num_seqs = config.max_num_seqs                     # 同一批次最多同步推进多少条请求
        self.max_num_batched_tokens = config.max_num_batched_tokens # 同一批次所有请求合起来最多不能超过多少tokens
        self.max_model_len = config.max_model_len                   # 单条 Sequence 最多允许多长的总上下文
        self.eos_token_id = config.eos_token_id
        self.speculative = bool(config.speculative)  # [MOD]

        # waiting：已经进入推理引擎，但还没有获得运行资格的请求。
        # running：已经被 Scheduler 选中，正在参与 prefill / decode 的请求。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
        )

    def add(self, seq: Sequence): # 新请求首先进入 waiting queue
        self.waiting.append(seq)

    def is_finished(self) -> bool: # waiting 和 running 都为空，说明整个引擎没有未完成请求
        return not self.waiting and not self.running

    def describe_output(output): # for debug only
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

    def check_consistency(self): # for debug only
        waiting_ids = {s.seq_id for s in self.waiting}
        running_ids = {s.seq_id for s in self.running}
        assert waiting_ids.isdisjoint(running_ids)
        for seq in self.waiting:
            assert seq.status is SequenceStatus.WAITING
            assert seq.num_computed_tokens <= seq.num_prompt_tokens
        for seq in self.running:
            assert seq.status is SequenceStatus.RUNNING
            assert seq.prompt_computed



    # 优先prefill, 其次decode, 且一个batch里面仅有prefill或者decode
    def schedule_separated(self) -> tuple[list[Sequence], bool]:
        if trace_enabled("TINYINFER_TRACE_SCHED"):
            print(
                "[sched]",
                "waiting=", len(self.waiting),
                "waiting_ids=", [s.seq_id for s in self.waiting],
                "running=", len(self.running),
                "running_ids=", [s.seq_id for s in self.running],
            )
        if trace_enabled("TINYINFER_CHECK_KV"):
            self.block_manager.check_consistency()

        scheduled: list[Sequence] = []
        token_budget = self.max_num_batched_tokens # 一轮调度中最多支持多少tokens
        # ------------------------------------------------------------
        # Phase A: admit waiting requests for prefill
        # ------------------------------------------------------------
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0] # 所有需要prefill的seq必然位于waiting队列中, 这里采用的是FIFO队列, 建模比较简单, 后续要改调度方案
            prefill_tokens = seq.num_tokens - seq.num_prefix_cached_tokens # 真正需要prefill的token数, 需要减去已经位于prefix cache中的token数目
            if prefill_tokens > token_budget:           # 确保本轮推理预算足够
                break
            if not self.block_manager.try_allocate_with_prefix_cache(seq): # 为本次请求分配prefix cache, 更新seq.block_table和blockManager中的block池
                break
            self.waiting.popleft() # 通过检查, 放入running列表
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = True
            seq.mark_scheduled(prefill_tokens)
            self.running.append(seq)
            scheduled.append(seq)
            token_budget -= prefill_tokens # 从总budget中扣除已经用掉的部分
        if scheduled:
            return scheduled, True
        # ------------------------------------------------------------
        # Phase B: one-token decode for running requests
        # ------------------------------------------------------------
        for seq in list(self.running): # 只有当本轮没有成功调度任何 prefill 时，才会走到 decode
            if token_budget <= 0:
                break
            # 已达到最大模型长度，不应该再送进 ModelRunner
            if seq.num_tokens >= self.max_model_len:
                continue
            seq.is_prefill = False
            seq.mark_scheduled(1)
            scheduled.append(seq)
            token_budget -= 1
        return scheduled, False


    # 当前的schedule会额外负责每个seq请求后续block的分配: 
    #  对于prefill会预分配所有prompt token对应的block(不论admit时是否已经在prefix cache中)
    #  对于decode, 如果有需求, 会预分配下一个block
    def schedule(self) -> SchedulerOutput:
        items: list[ScheduledItem] = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # waiting非空时，Phase A至少给prefill预留1个seq预算
        decode_seq_budget = seq_budget if not self.waiting else max(0, seq_budget - 1)

        # Phase A: decode first with round-robin; 注意当前通常num_computed_token比num_tokens少1, 因为有1个token是上一轮decode得到的、尚未算出KV cache
        running_count = len(self.running)
        for _ in range(running_count):
            if token_budget <= 0 or decode_seq_budget <= 0:  # 只要waiting-list不为空, 则至少预留一个slot给prefill
                break
            
            seq = self.running[0]
            if seq.num_tokens >= self.max_model_len: # 本条seq的token总数已经达到上限;
                self.running.rotate(-1)
                continue
            if not seq.needs_decode: # 确认当前需要继续decode
                self.running.rotate(-1)
                continue

            start = seq.num_computed_tokens
            if not self.block_manager.ensure_capacity_for_token_position(seq, start): # 预分配block
                self.running.rotate(-1)
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
            decode_seq_budget -= 1

            # 最近被decode调度过的seq放到running尾部
            self.running.rotate(-1)

        # Phase B: then chunked prefill with remaining budget; 注意只有一个seq的chunked prefill彻底完成, 其才会离开waiting list
        waiting_count = len(self.waiting)
        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            seq = self.waiting[0] # 从头部获取第一个waiting的seq

            # First admission: prefix lookup + block allocation.
            if not seq.block_table: # block_table为空说明必然为首次进行prefill, 因此需要admit
                if not self.block_manager.try_admit_with_prefix_cache(seq):
                    self.waiting.rotate(-1)
                    continue

            remaining = seq.num_prompt_tokens - seq.num_computed_tokens
            if remaining <= 0:
                raise RuntimeError("waiting prefill has no remaining prompt tokens")

            chunk = min(remaining, token_budget) # 这次chunk的大小
            start = seq.num_computed_tokens
            end = start + chunk

            sample_after = end == seq.num_prompt_tokens
            seq.mark_scheduled(chunk) # 开始调度一刻

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

            # partial prefill 做 round-robin优先级调度, 最近被处理过的seq被放到 waiting 尾部
            self.waiting.rotate(-1)

        return SchedulerOutput(items)


    def postprocess_speculative(self, output: SchedulerOutput, results, rollback_cache_tail):  # [MOD]
        """Commit only the accepted prefix; rejected speculative suffixes are invalidated."""  # [MOD]
        finished = []  # [MOD]
        for item in output.items:  # [MOD]
            if item.is_prefill:  # [MOD]
                raise ValueError("postprocess_speculative accepts decode items only")  # [MOD]
            seq = item.seq  # [MOD]
            result = results.get(seq.seq_id)  # [MOD]
            if result is None or not result.token_ids:  # [MOD]
                raise RuntimeError("speculative decode returned no tokens")  # [MOD]

            appended = 0  # [MOD]
            for token_id in result.token_ids:  # [MOD]
                seq.append_token(token_id)  # [MOD]
                appended += 1  # [MOD]
                if seq.should_stop(self.eos_token_id, self.max_model_len):  # [MOD]
                    break  # [MOD]

            accepted_emitted = min(result.accepted_count, appended)  # [MOD]
            keep_tokens = item.start_pos + 1 + accepted_emitted  # [MOD]
            seq.num_computed_tokens = keep_tokens  # [MOD]
            seq.num_scheduled_tokens = 0  # [MOD]

            # Target and Draft can have different physical write frontiers during a round. # [MOD]
            # Roll each storage back to keep_tokens, but keep every reserved block ID. # [MOD]
            rollback_cache_tail(  # [MOD]
                seq, keep_tokens, result.verified_end, result.draft_written_end  # [MOD]
            )  # [MOD]
            # Registration happens only here, after both frontiers have been rolled back/aligned. # [MOD]
            self.block_manager.cache_computed_full_blocks(  # [MOD]
                seq, upto_token=seq.num_computed_tokens, require_draft=True  # [MOD]
            )  # [MOD]

            if seq.should_stop(self.eos_token_id, self.max_model_len):  # [MOD]
                seq.mark_finished()  # [MOD]
                finished.append(seq)  # [MOD]

        if finished:  # [MOD]
            done_ids = {seq.seq_id for seq in finished}  # [MOD]
            self.running = deque(seq for seq in self.running if seq.seq_id not in done_ids)  # [MOD]
            for seq in finished:  # [MOD]
                self.block_manager.free(seq)  # [MOD]

    def postprocess(self, output: SchedulerOutput, sampled_tokens: dict[int, int]):
        finished = []
        prefill_completed = []

        for item in output.items:
            seq = item.seq
        
            # 1. 本轮 scheduled token 的 KV 已经完成
            seq.num_computed_tokens += item.num_tokens
            seq.num_scheduled_tokens = 0

            # 2. 到新的 computed frontier 为止，注册 newly-full blocks
            self.block_manager.cache_computed_full_blocks(
                seq,
                upto_token=seq.num_computed_tokens,
                require_draft=self.speculative,  # [MOD] Prefill in SD mode must have completed both model KV writes.
            )

            # 3. partial prefill 不采样，直接结束本 item
            if not item.sample_after: # 此时sample_tokens实际为None
                continue
            else:
                token_id = sampled_tokens[seq.seq_id]
                seq.append_token(token_id) # 这里让num_tokens + 1, 但num_computed_tokens没变

            # 4. 采样后检查停止条件
            if seq.should_stop(self.eos_token_id, self.max_model_len):
                #seq.status = SequenceStatus.FINISHED
                seq.mark_finished()
                finished.append(seq)
                continue

            # 5. 如果刚完成 prompt，则进入 running decode 集合
            if item.is_prefill: # 走到这里的prefill必然是final prefill
                prefill_completed.append(seq)
                seq.status = SequenceStatus.RUNNING

        # waiting 中删除完成 prefill 或 finished 的请求。
        remove_ids = { seq.seq_id for seq in prefill_completed + finished}
        if remove_ids: # waiting list中仅保留原本存在且当前不在remove_ids中的seq
            self.waiting = deque(seq for seq in self.waiting if seq.seq_id not in remove_ids)

        # 新完成 prefill 的 seq 开始进入 running decode
        for seq in prefill_completed:
            self.running.append(seq)

        # running 中删除 finished
        if finished:
            done_ids = {seq.seq_id for seq in finished}
            self.running = deque(seq for seq in self.running if seq.seq_id not in done_ids)

        # 最后释放 request ownership；persistent cached block 不会被 reset。
        for seq in finished:
            self.block_manager.free(seq)
```

## Step 10：在 LLMEngine 中增量分流，不破坏 baseline

`Config.speculative=False` 时，`step_with_updates()` 仍然走原来的 `model_runner.run(output)` + `scheduler.postprocess()`。

SD 模式下，如果一轮 scheduler 同时选出 prefill 与 decode items，则把它们拆成两个子批次：prefill 子批次继续走原 scheduler postprocess，但 `ModelRunner` 同时填充 Draft KV；decode 子批次走 `run_speculative()` 和 `postprocess_speculative()`。UI 的 `StreamUpdate` 仍然发完整 completion token 列表，因此一轮接受多个 token 时无需更改前端的文本拼接协议。

初始化时还会比较 Target 与 Draft tokenizer 的 token-to-ID 映射以及 EOS ID。仅仅 `vocab_size` 相同并不能证明 token ID 语义一致。

**完整文件：** `tinyinfer/engine/llm_engine.py`

```python
from dataclasses import dataclass
import torch

from transformers import AutoTokenizer
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence, SchedulerOutput  # [MOD]
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
        if self.config.speculative:  # [MOD]
            draft_tokenizer = AutoTokenizer.from_pretrained(  # [MOD]
                self.config.draft_model, trust_remote_code=True  # [MOD]
            )  # [MOD]
            if self.tokenizer.get_vocab() != draft_tokenizer.get_vocab():  # [MOD]
                raise ValueError(  # [MOD]
                    "target and draft tokenizers do not share the same token-to-ID mapping"  # [MOD]
                )  # [MOD]
            if draft_tokenizer.eos_token_id != self.tokenizer.eos_token_id:  # [MOD]
                raise ValueError("target and draft EOS token IDs differ")  # [MOD]


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
        output = self.scheduler.schedule()  # [MOD]
        if not output.items:
            return [], []

        if not self.config.speculative:  # [MOD]
            sampled_tokens = self.model_runner.run(output)  # [MOD]
            self.scheduler.postprocess(output, sampled_tokens)  # [MOD]
        else:  # [MOD]
            # Prefill retains the original scheduler semantics. Draft KV is filled  # [MOD]
            # alongside target KV, then decode items enter the speculative path.  # [MOD]
            prefill_items = [item for item in output.items if item.is_prefill]  # [MOD]
            decode_items = [item for item in output.items if not item.is_prefill]  # [MOD]
            sampled_tokens = {}  # [MOD]
            if prefill_items:  # [MOD]
                prefill_output = SchedulerOutput(prefill_items)  # [MOD]
                prefill_tokens = self.model_runner.run_prefill_with_draft(prefill_output)  # [MOD]
                self.scheduler.postprocess(prefill_output, prefill_tokens)  # [MOD]
                sampled_tokens.update(prefill_tokens)  # [MOD]
            if decode_items:  # [MOD]
                decode_output = SchedulerOutput(decode_items)  # [MOD]
                results = self.model_runner.run_speculative(  # [MOD]
                    decode_output, self.scheduler.block_manager  # [MOD]
                )  # [MOD]
                self.scheduler.postprocess_speculative(  # [MOD]
                    decode_output, results, self.model_runner.rollback_cache_tail  # [MOD]
                )  # [MOD]
                sampled_tokens.update({  # [MOD]
                    seq_id: result.token_ids[-1]  # [MOD]
                    for seq_id, result in results.items()  # [MOD]
                    if result.token_ids  # [MOD]
                })  # [MOD]

        # The UI consumes the complete completion prefix, so one SD round may  # [MOD]
        # legitimately publish several newly generated tokens in one update.  # [MOD]
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

    
    def set_max_draft_tokens(self, value: int) -> None:  # [MOD]
        if not self.config.speculative:  # [MOD]
            raise RuntimeError("speculative decoding is not enabled")  # [MOD]
        self.model_runner.set_max_draft_tokens(value)  # [MOD]

    def speculative_snapshot(self) -> dict:  # [MOD]
        return self.model_runner.speculative_snapshot()  # [MOD]

    # 资源快照只读取 CPU metadata，不做 CUDA synchronize; 仅获取resource快照
    def runtime_resource_snapshot(self) -> dict:
        profile = self.model_runner.resource_profile
        if profile is None:  
            return {  
                "total_bytes": 0,  
                "reserved_bytes": 0,  
                "weights_bytes": 0, 
                "temporary_peak_bytes": 0,  
                "allocated_kv_bytes": 0, 
                "available_kv_bytes": 0,  
                "num_blocks": 0,  
                "active_block_ids": [],  
                "cached_block_ids": [],  
            }  
        blocks = self.scheduler.block_manager.blocks
        active_block_ids = [block.block_id for block in blocks if block.ref_count > 0]  # [MOD]
        cached_block_ids = [block.block_id for block in blocks if block.ref_count == 0 and block.hash is not None]  # [MOD]
        # Each logical block ID maps to one target block and (in SD mode) one  # [MOD]
        # draft block. The resource bar accounts for both, while the square grid  # [MOD]
        # visualizes only target logical block IDs.  # [MOD]
        allocated_kv_bytes = len(active_block_ids) * profile.block_bytes  # [MOD]
        kv_pool_bytes = profile.num_blocks * profile.block_bytes  # [MOD]
        available_kv_bytes = max(  # [MOD]
            0,  # [MOD]
            profile.total_bytes - profile.reserved_bytes - profile.weights_bytes  # [MOD]
            - profile.temporary_peak_bytes - allocated_kv_bytes,  # [MOD]
        )  # [MOD]
        return {
            "total_bytes": profile.total_bytes,  # [MOD]
            "reserved_bytes": profile.reserved_bytes,  # [MOD]
            "weights_bytes": profile.weights_bytes,  # [MOD]
            "target_weights_bytes": profile.target_weights_bytes,  # [MOD]
            "draft_weights_bytes": profile.draft_weights_bytes,  # [MOD]
            "temporary_peak_bytes": profile.temporary_peak_bytes,  # [MOD]
            "allocated_kv_bytes": allocated_kv_bytes,  # [MOD]
            "available_kv_bytes": available_kv_bytes,  # [MOD]
            "kv_pool_bytes": kv_pool_bytes,  # [MOD]
            "block_bytes": profile.target_block_bytes or profile.block_bytes,  # [MOD]
            "target_block_bytes": profile.target_block_bytes or profile.block_bytes,  # [MOD]
            "draft_block_bytes": profile.draft_block_bytes,  # [MOD]
            "combined_block_bytes": profile.block_bytes,  # [MOD]
            "num_blocks": len(blocks),
            "active_block_ids": active_block_ids,  # [MOD]
            "cached_block_ids": cached_block_ids,  # [MOD]
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

## Step 11：把全局 SD 统计与 Max Draft Tokens 暴露给 Runtime Service

Runtime snapshot 的 `speculative` 字段保留从引擎启动开始累计的全局统计：`max_draft_tokens`、`acceptance_rate` 与 `avg_acceptance_length`。其中 Acceptance Rate = 累计 accepted proposal tokens / 累计 proposed tokens；Avg Acceptance Length = 累计 accepted proposal tokens / 累计有效 SD decode request-rounds。

`max_draft_tokens` 是整个 ModelRunner 的全局参数，所有用户共享同一个值。本讲不再做 EMA / adaptive K。为了避免某个正在执行的 batch 读到运行时被改写的 K，Runtime Service 只允许在 `_busy_users` 为空、也就是 5 个 User 都处于 Idle 时修改它；前端 slider 会禁用，后端 `configure_speculative()` 也会再次检查，形成双重保护。

**完整文件：** `tinyinfer/runtime/dynamic_engine.py`

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
    speculative: dict | None = None  # [MOD]
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
        self._cache_hit_rate_sum = 0.0 

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

    def all_users_idle(self) -> bool:  # [MOD]
        with self._busy_lock:  # [MOD]
            return not self._busy_users  # [MOD]

    def configure_speculative(self, max_draft_tokens: int) -> dict:  # [MOD]
        if not self.engine.config.speculative:  # [MOD]
            raise RuntimeError("this server was started without --speculative")  # [MOD]
        # Hold the busy lock through the tiny config update so submit_many() cannot  # [MOD]
        # race between the idle check and set_max_draft_tokens().  # [MOD]
        with self._busy_lock:  # [MOD]
            if self._busy_users:  # [MOD]
                raise RuntimeError(  # [MOD]
                    "max_draft_tokens can only be changed when all users are idle"  # [MOD]
                )  # [MOD]
            self.engine.set_max_draft_tokens(max_draft_tokens)  # [MOD]
        return self._speculative_snapshot()  # [MOD]

    def _speculative_snapshot(self) -> dict:  # [MOD]
        snapshot = self.engine.speculative_snapshot()  # [MOD]
        snapshot["all_users_idle"] = self.all_users_idle()  # [MOD]
        return snapshot  # [MOD]

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
        self._cache_hit_rate_sum += float(statistics.get("cache_hit_rate", 0.0)) 
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
            "cache_hit_rate": self._cache_hit_rate_sum / self._completed_requests if self._completed_requests else None,
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
            "speculative": self._speculative_snapshot(),  # [MOD]
        }
        with self._snapshot_lock:
            self._last_runtime_snapshot = deepcopy(snapshot)

        self._emit(
            RuntimeEvent(
                type="runtime",
                resource=snapshot["resource"],
                performance=snapshot["performance"],
                speculative=snapshot["speculative"],  # [MOD]
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
                    self._emit_runtime_snapshot(force=True)
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
                self._emit_runtime_snapshot(force=True)
            except Exception as exc:
                self._fail_active_requests(str(exc))
                self._emit_runtime_snapshot(force=True)
                return
```

## Step 12：WebSocket 接收全局 Max Draft Tokens 配置消息

前端只发送 `{type: "configure_sd", max_draft_tokens}`。后端除了检查数值范围，还由 Runtime Service 强制检查当前是否所有 User 都空闲；只要有任意请求已经提交或正在运行，就拒绝本次修改。成功后返回 `sd_config` 确认消息。控制器在 baseline 模式下隐藏，避免影响原本 UI 的使用方式。

**完整文件：** `tinyinfer/ui/app.py`

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tinyinfer.runtime.dynamic_engine import (
    DynamicEngineService,
    SubmitRequest,
)
from tinyinfer.sampling_params import SamplingParams


STATIC_DIR = Path(__file__).with_name("static")


class SocketHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.events: asyncio.Queue[dict] | None = None

    def bind_loop(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.events = asyncio.Queue()

    def publish_from_thread(self, event: dict) -> None:
        if self.loop is None or self.events is None:
            return
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    async def broadcast_loop(self) -> None:
        assert self.events is not None
        while True:
            event = await self.events.get()
            dead = []
            for socket in self.clients:
                try:
                    await socket.send_json(event)
                except Exception:
                    dead.append(socket)
            for socket in dead:
                self.clients.discard(socket)


def _parse_sampling_params(
    item: dict,
    default_max_tokens: int,
    default_temperature: float,
    default_greedy: bool,
) -> SamplingParams:
    max_tokens = int(item.get("max_tokens", default_max_tokens))
    temperature = float(item.get("temperature", default_temperature))
    greedy = bool(item.get("greedy", default_greedy))

    if not 32 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be in [32, 8192]")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be in [0.1, 2.0]")

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        greedy=greedy,
    )


def create_app(
    service: DynamicEngineService,
    max_tokens: int = 128,
    temperature: float = 0.8,
    greedy: bool = True,
) -> FastAPI:
    hub = SocketHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop()
        broadcaster = asyncio.create_task(hub.broadcast_loop())
        service.start(hub.publish_from_thread) 
        try:
            yield
        finally:
            service.stop()
            broadcaster.cancel()

    app = FastAPI(title="tinyInfer UI", lifespan=lifespan) # 用FastAPI创建app
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static") # 指向构建网页app的static目录

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(socket: WebSocket):
        await socket.accept()
        hub.clients.add(socket)

        snapshot = service.current_runtime_snapshot()
        if snapshot is not None:
            await socket.send_json({"type": "runtime", **snapshot})

        try:
            while True:
                payload = await socket.receive_json()
                request_type = payload.get("type")
                if request_type == "configure_sd":  # [MOD]
                    try:  # [MOD]
                        raw_max = payload.get("max_draft_tokens")  # [MOD]
                        if raw_max is None:  # [MOD]
                            raise ValueError("max_draft_tokens is required")  # [MOD]
                        snapshot = service.configure_speculative(int(raw_max))  # [MOD]
                        await socket.send_json({"type": "sd_config", "speculative": snapshot})  # [MOD]
                    except (TypeError, ValueError, RuntimeError) as exc:  # [MOD]
                        await socket.send_json({"type": "error", "message": str(exc)})  # [MOD]
                    continue  # [MOD]
                raw_requests = (
                    payload.get("requests", [])
                    if request_type == "submit_many"
                    else [payload]
                )

                requests = []
                requested_users = []
                for item in raw_requests:
                    user_id = str(item.get("user_id", ""))
                    text = str(item.get("text", ""))
                    if not user_id or not text.strip():
                        continue
                    requested_users.append(user_id)

                    try:
                        params = _parse_sampling_params(
                            item,
                            max_tokens,
                            temperature,
                            greedy,
                        )
                    except (TypeError, ValueError) as exc:
                        await socket.send_json(
                            {
                                "type": "error",
                                "user_id": user_id,
                                "message": str(exc),
                            }
                        )
                        continue

                    requests.append(
                        SubmitRequest(
                            user_id=user_id,
                            text=text,
                            params=params,
                        )
                    )

                accepted = service.submit_many(requests)
                await socket.send_json(
                    {
                        "type": "accepted",
                        "accepted_users": accepted,
                        "requested_users": requested_users,
                    }
                )
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(socket)

    return app
```

## Step 13：保留布局并添加全局 SD 指标与 Draft 长度控制

HTML 保留原来的 Users、Statistics、Controller 三个主要区域。资源条仍然是五段式；新增权重拆分说明、双模型 paired KV pool 的字节说明。`Performance statistics` 的 `Cache Hit Rate` 子栏正下方增加 `Speculative Decoding` 子栏，**横向并排只显示三个全局参数：Max Draft Tokens、Acceptance Rate、Avg Acceptance Length**。

Draft 长度 slider 放在右侧 Controller 的 `Start all ready` 按钮右侧，范围固定为 2～10。它是整个 Engine 的全局参数，因此不在每个 User panel 中重复设置。

**完整文件：** `tinyinfer/ui/static/index.html`

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tinyInfer Console</title>
  <link rel="stylesheet" href="/static/style.css?v=advance07">  <!-- # [MOD] -->
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
          <div id="mem-model-breakdown" class="subtle memory-breakdown">Model weights: --</div> <!-- # [MOD] -->

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
          <div id="sd-metrics" class="sd-metrics" hidden> <!-- # [MOD] -->
            <div class="percentile-head">Speculative Decoding</div> <!-- # [MOD] -->
            <div class="percentile-grid sd-metric-grid"> <!-- # [MOD] -->
              <div class="metric-card"><span>Max Draft Tokens</span><strong id="sd-max-draft-stat">--</strong></div> <!-- # [MOD] -->
              <div class="metric-card"><span>Acceptance Rate</span><strong id="sd-acceptance-rate">--</strong></div> <!-- # [MOD] -->
              <div class="metric-card"><span>Avg Acceptance Length</span><strong id="sd-acceptance-length">--</strong></div> <!-- # [MOD] -->
            </div> <!-- # [MOD] -->
          </div> <!-- # [MOD] -->
        </section>
      </section>

      <section class="panel controller">
        <div class="panel-title">Controller</div>
        <div class="controller-grid">
          <button id="start-all" class="primary" disabled>Start all ready</button>
          <div id="sd-controls" class="sd-controls" hidden> <!-- # [MOD] -->
            <div class="control-row sd-draft-control">  <!-- # [MOD] -->
              <span>Max Draft Tokens</span>  <!-- # [MOD] -->
              <input id="sd-max-draft" type="range" min="2" max="10" step="1" value="5">  <!-- # [MOD] -->
              <span id="sd-max-draft-value" class="control-value">5</span>  <!-- # [MOD] -->
            </div>  <!-- # [MOD] -->
          </div> <!-- # [MOD] -->
          <div class="controller-slot">future</div>
        </div>
      </section>
    </aside>
  </main>

  <script src="/static/app.js?v=advance07"></script>  <!-- # [MOD] -->
</body>
</html>
```

## Step 14：前端渲染双 cache 预算、全局 SD 统计与只读/可调状态

前端只绘制一套逻辑 block ID 方格；每个方格对应相同 block ID 的 Target block 与 Draft block。资源条的 `Weights` 是两模型权重之和，`Allocated For KV` 按 active logical blocks 乘以两份 KV 的合计成本。UI 会同时显示整个 paired pool 的预分配字节数，避免把逻辑 active bytes 误认为物理 pool 的全部大小。

SD 部分只展示三个全局指标。slider 的本地 `disabled` 状态同时参考 5 个 User panel 的 `busy` 状态和后端 snapshot 的 `all_users_idle`；任何一个 User 进入 Running，slider 立即禁用。后端仍会再次检查，因此前端状态出现竞态也不会在请求执行中修改全局 K。

**完整文件：** `tinyinfer/ui/static/app.js`

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
const sdControls = document.querySelector("#sd-controls"); // # [MOD]
const sdMetrics = document.querySelector("#sd-metrics"); // # [MOD]
const sdMaxDraft = document.querySelector("#sd-max-draft"); // # [MOD]
const sdMaxDraftValue = document.querySelector("#sd-max-draft-value"); // # [MOD]
const ws = new WebSocket(`ws://${location.host}/ws`);

let sdAllUsersIdle = true; // # [MOD] 后端 snapshot 给出的全局空闲状态
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

  // Global Max Draft Tokens may only change when every user is idle. // # [MOD]
  const allLocalUsersIdle = [...states.values()].every((state) => !state.busy); // # [MOD]
  sdMaxDraft.disabled = // # [MOD]
    ws.readyState !== WebSocket.OPEN || // # [MOD]
    sdControls.hidden || // # [MOD]
    !allLocalUsersIdle || // # [MOD]
    !sdAllUsersIdle; // # [MOD]
  sdMaxDraft.title = sdMaxDraft.disabled // # [MOD]
    ? "Max Draft Tokens can only be changed when all users are idle." // # [MOD]
    : "Global Max Draft Tokens for all users."; // # [MOD]
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
  const targetWeights = resource.target_weights_bytes ?? weights; // # [MOD]
  const draftWeights = resource.draft_weights_bytes ?? 0; // # [MOD]
  document.querySelector("#mem-model-breakdown").textContent = draftWeights > 0 // # [MOD]
    ? `Target weights ${formatBytes(targetWeights)} + draft weights ${formatBytes(draftWeights)}; KV pools are preallocated.` // # [MOD]
    : `Target weights ${formatBytes(targetWeights)}; KV pool is preallocated.`; // # [MOD]

  setSegment("#segment-reserved", reserved, total);
  setSegment("#segment-weights", weights, total); // # [MOD]
  setSegment("#segment-temporary", temporary, total); // # [MOD]
  setSegment("#segment-allocated", allocated, total);
  setSegment("#segment-available", available, total);

  const numBlocks = resource.num_blocks ?? 0;
  const activeIds = resource.active_block_ids ?? [];
  const cachedIds = resource.cached_block_ids ?? []; // [MOD]
  // [MOD] 数量统计和单block容量拆成两行，避免挤压右侧三态图例。
  const poolLabel = (resource.draft_block_bytes ?? 0) > 0 ? "paired pool" : "pool"; // # [MOD]
  document.querySelector("#kv-summary").textContent = // # [MOD]
    `${activeIds.length} active · ${cachedIds.length} cached · ${numBlocks} total · ${formatBytes(resource.kv_pool_bytes ?? 0)} ${poolLabel}`; // # [MOD]
  const targetBlockBytes = resource.target_block_bytes ?? resource.block_bytes ?? 0; // # [MOD]
  const draftBlockBytes = resource.draft_block_bytes ?? 0; // # [MOD]
  document.querySelector("#kv-block-size").textContent = draftBlockBytes > 0 // # [MOD]
    ? `${formatBytes(targetBlockBytes)} target + ${formatBytes(draftBlockBytes)} draft / logical block` // # [MOD]
    : `${formatBytes(targetBlockBytes)} / block`; // # [MOD]
  updateBlockGrid(numBlocks, activeIds, cachedIds);
}

function updateSpeculative(speculative) { // # [MOD]
  if (!speculative) return; // # [MOD]
  const enabled = Boolean(speculative.enabled); // # [MOD]
  sdControls.hidden = !enabled; // # [MOD]
  sdMetrics.hidden = !enabled; // # [MOD]
  if (!enabled) { // # [MOD]
    refreshControls(); // # [MOD]
    return; // # [MOD]
  } // # [MOD]

  sdAllUsersIdle = Boolean(speculative.all_users_idle); // # [MOD]
  if (document.activeElement !== sdMaxDraft) { // # [MOD]
    sdMaxDraft.value = String(speculative.max_draft_tokens ?? 5); // # [MOD]
  } // # [MOD]
  sdMaxDraftValue.textContent = sdMaxDraft.value; // # [MOD]
  document.querySelector("#sd-max-draft-stat").textContent = String(speculative.max_draft_tokens ?? "--"); // # [MOD]
  document.querySelector("#sd-acceptance-rate").textContent = formatRate(speculative.acceptance_rate); // # [MOD]
  const avgLength = speculative.avg_acceptance_length; // # [MOD]
  document.querySelector("#sd-acceptance-length").textContent = // # [MOD]
    avgLength == null ? "--" : avgLength.toFixed(2); // # [MOD]
  refreshControls(); // # [MOD]
} // # [MOD]

function sendSpeculativeConfig() { // # [MOD]
  const allLocalUsersIdle = [...states.values()].every((state) => !state.busy); // # [MOD]
  if ( // # [MOD]
    ws.readyState !== WebSocket.OPEN || // # [MOD]
    sdControls.hidden || // # [MOD]
    !allLocalUsersIdle || // # [MOD]
    !sdAllUsersIdle // # [MOD]
  ) return; // # [MOD]

  ws.send(JSON.stringify({ // # [MOD]
    type: "configure_sd", // # [MOD]
    max_draft_tokens: Number(sdMaxDraft.value), // # [MOD]
  })); // # [MOD]
} // # [MOD]

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
  //document.querySelector("#performance-cache-hit-rate").textContent = formatRate(performance.cache_hit_rate); // # [MOD]
  const cacheNode = document.querySelector("#performance-cache-hit-rate");
  if (cacheNode) {
    cacheNode.textContent = formatRate(performance.cache_hit_rate);
  }
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
    updateSpeculative(event.speculative); // # [MOD]
    return;
  }
  if (event.type === "sd_config") { // # [MOD]
    updateSpeculative(event.speculative); // # [MOD]
    return; // # [MOD]
  } // # [MOD]

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
  if (!state) { // # [MOD]
    if (event.type === "error" && event.message) { // # [MOD]
      console.error(event.message); // # [MOD]
    } // # [MOD]
    return; // # [MOD]
  } // # [MOD]

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

sdMaxDraft.addEventListener("input", () => { // # [MOD]
  sdMaxDraftValue.textContent = sdMaxDraft.value; // # [MOD]
}); // # [MOD]
sdMaxDraft.addEventListener("change", sendSpeculativeConfig); // # [MOD]

ws.addEventListener("open", refreshControls);
ws.addEventListener("close", () => {
  for (const state of states.values()) state.button.disabled = true;
  startAll.disabled = true;
  sdMaxDraft.disabled = true; // # [MOD]
});

USERS.forEach(makePanel);
initPerformanceGrid();
refreshControls();
```

## Step 15：SD 统计与全局 Draft slider 样式

`sd-metric-grid` 固定为三列，因此 Max Draft Tokens / Acceptance Rate / Avg Acceptance Length 横向并排。Controller 中 `sd-controls` 占两个 grid column，紧跟在 `Start all ready` 右侧，为 slider 留出足够宽度；baseline 模式下这些节点仍由 `hidden` 隐藏。

**完整文件：** `tinyinfer/ui/static/style.css`

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
  --reserved: #6F737B;
  --weights: #4F7CAC;
  --temporary: #63A4FF;

  --allocated: #F59E42;
  --available: #4FAE36;
  
  --empty: #67C23A;
  --cached: #357A2D;
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
.block-empty { background: var(--empty); } 
.block-cached { background: var(--cached); } 
.block-active { background: var(--allocated); } 

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
.kv-block.empty { background: var(--empty); } /* [MOD] 从未分配 */
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
/* Advance 07: SD controls are only visible when the server enables SD. */  /* # [MOD] */
.sd-controls {  /* # [MOD] */
  grid-column: span 2;  /* # [MOD] Start All 右侧，占两列给 slider 留宽度。 */
  min-height: 39px;  /* # [MOD] */
  padding: 6px 8px;  /* # [MOD] */
  border: 1px solid #29303d;  /* # [MOD] */
  border-radius: 11px;  /* # [MOD] */
  background: #10151d;  /* # [MOD] */
}  /* # [MOD] */
.sd-controls[hidden],  /* # [MOD] */
.sd-metrics[hidden] {  /* # [MOD] */
  display: none !important;  /* # [MOD] */
}  /* # [MOD] */
.sd-draft-control {  /* # [MOD] */
  grid-template-columns: 88px minmax(0, 1fr) 28px;  /* # [MOD] */
  margin-top: 0;  /* # [MOD] */
}  /* # [MOD] */
.sd-metric-grid {  /* # [MOD] */
  grid-template-columns: repeat(3, minmax(0, 1fr));  /* # [MOD] */
}  /* # [MOD] */

.memory-breakdown {  /* # [MOD] */
  margin-top: 7px;  /* # [MOD] */
  line-height: 1.4;  /* # [MOD] */
}  /* # [MOD] */
```

## Step 16：新增统一启动入口，参数选择 baseline / SD

新增 `examples/advance07_speculative_decoding.py`。不传 `--speculative` 时，默认仍加载本地 Qwen3-0.6B，作为原始 baseline；传入 `--speculative` 时，默认 Target 为 Qwen3-4B，Draft 为 Qwen3-0.6B。可通过 `--model` 和 `--draft-model` 覆盖路径。旧的 `examples/advance02_ui_plus.py` 不需要修改。

**完整文件：** `examples/advance07_speculative_decoding.py`

```python
import argparse
from pathlib import Path

import torch
import uvicorn

from tinyinfer import LLM
from tinyinfer.config import Config
from tinyinfer.runtime import DynamicEngineService
from tinyinfer.ui import create_app


DEFAULT_MODEL_DIR = Path.home() / "huggingface"
DEFAULT_DRAFT_MODEL = DEFAULT_MODEL_DIR / "Qwen3-0.6B"
DEFAULT_SD_TARGET_MODEL = DEFAULT_MODEL_DIR / "Qwen3-4B"
DEFAULT_BASELINE_MODEL = DEFAULT_MODEL_DIR / "Qwen3-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tinyInfer in baseline or speculative-decoding mode."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target model path. Defaults to Qwen3-0.6B in baseline mode and Qwen3-4B in SD mode.",
    )
    parser.add_argument(
        "--draft-model",
        default=str(DEFAULT_DRAFT_MODEL),
        help="Draft model path used only with --speculative.",
    )
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding.")
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        choices=range(2, 11),
        default=5,
        help="Global maximum draft length shared by all users; allowed range: 2..10.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    target_model = args.model
    if target_model is None:
        target_model = str(DEFAULT_SD_TARGET_MODEL if args.speculative else DEFAULT_BASELINE_MODEL)

    config = Config(
        model=target_model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
        speculative=args.speculative,
        draft_model=args.draft_model if args.speculative else None,
        max_draft_tokens=args.max_draft_tokens,
    )
    engine = LLM(config, device="cuda", dtype=torch.bfloat16)
    service = DynamicEngineService(engine, snapshot_interval_s=0.25)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

## Step 17：新增 SD helper 单元测试

这些测试不依赖真实模型权重，也不要求 CUDA：验证 top-k 概率、相同分布下的接受、拒绝时 residual 的支持集，以及 Target/Draft 两份 cache 只清除未提交后缀。完整模型的正确性与性能还需要在本地 CUDA 环境做 smoke test。

**完整文件：** `tests/test_speculative_decoding.py`

```python
import torch

from tinyinfer.engine.speculative import (
    clear_kv_cache_tail,
    distribution_from_logits,
    sample_residual_correction,
    speculative_accept_or_correct,
    speculative_acceptance_mask,
)
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.engine.block_manager import BlockManager  # [MOD]
from tinyinfer.engine.sequence import Sequence  # [MOD]
from tinyinfer.sampling_params import SamplingParams  # [MOD]


def test_distribution_from_logits_applies_temperature_and_top_k():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    probs = distribution_from_logits(logits, temperature=1.0, top_k=2)
    assert probs.shape == logits.shape
    assert torch.allclose(probs[:3], torch.zeros(3))
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert probs[4] > probs[3] > 0


def test_speculative_accepts_when_distributions_match():
    probs = torch.tensor([0.25, 0.75])
    accepted, token = speculative_accept_or_correct(1, probs, probs)
    assert accepted is True
    assert token == 1


def test_speculative_rejection_samples_from_positive_residual():
    target = torch.tensor([0.1, 0.9])
    draft = torch.tensor([0.8, 0.2])
    torch.manual_seed(123)
    accepted, token = speculative_accept_or_correct(0, target, draft)
    assert accepted is False
    # max(target - draft, 0) has support only on token 1.
    assert token == 1


def test_rollback_clears_each_cache_to_its_own_written_frontier():
    target = PagedKVCache(
        num_layers=2, num_blocks=2, block_size=4, num_kv_heads=1,
        head_dim=2, dtype=torch.float32, device="cpu",
    )
    draft = PagedKVCache(
        num_layers=1, num_blocks=2, block_size=4, num_kv_heads=1,
        head_dim=2, dtype=torch.float32, device="cpu",
    )
    target.storage.fill_(1.0)
    draft.storage.fill_(1.0)

    # Logical blocks [0, 1] map to physical blocks [1, 0]. Keep tokens [0, 5).
    # Target wrote positions 5, 6; Draft wrote only position 5. Each cache is
    # cleared to its own exclusive written frontier, without touching committed slots.
    clear_kv_cache_tail(
        [target], block_table=[1, 0], block_size=4,
        keep_tokens=5, written_end=7,
    )
    clear_kv_cache_tail(
        [draft], block_table=[1, 0], block_size=4,
        keep_tokens=5, written_end=6,
    )
    assert torch.all(target.storage[:, :, 1, :, :, :] == 1.0)
    assert torch.all(draft.storage[:, :, 1, :, :, :] == 1.0)
    for cache in (target, draft):
        assert torch.all(cache.storage[:, :, 0, 0, :, :] == 1.0)
        assert torch.all(cache.storage[:, :, 0, 1, :, :] == 0.0)
        assert torch.all(cache.storage[:, :, 0, 3, :, :] == 1.0)
    assert torch.all(target.storage[:, :, 0, 2, :, :] == 0.0)
    assert torch.all(draft.storage[:, :, 0, 2, :, :] == 1.0)





def test_prefix_cache_registers_only_blocks_complete_in_both_models():
    seq = Sequence(list(range(8)), SamplingParams())
    seq.block_size = 4
    block_manager = BlockManager(num_blocks=4, block_size=4)
    assert block_manager.ensure_capacity_for_token_position(seq, 7)

    seq.target_kv_computed_tokens = 8
    seq.draft_kv_computed_tokens = 4
    block_manager.cache_computed_full_blocks(seq, upto_token=8, require_draft=True)
    first_block, second_block = seq.block_table[:2]
    assert block_manager.blocks[first_block].hash is not None
    assert block_manager.blocks[second_block].hash is None

    seq.draft_kv_computed_tokens = 8
    block_manager.cache_computed_full_blocks(seq, upto_token=8, require_draft=True)
    assert block_manager.blocks[second_block].hash is not None


def test_reserved_block_ids_survive_speculative_rollback():
    seq = Sequence(list(range(12)), SamplingParams())
    seq.block_size = 4
    block_manager = BlockManager(num_blocks=4, block_size=4)
    assert block_manager.ensure_capacity_for_token_position(seq, 8)
    reserved_ids = list(seq.block_table)
    assert len(reserved_ids) == 3

    # Simulate logical rollback. The reservation table intentionally does not shrink.
    seq.num_computed_tokens = 3
    seq.target_kv_computed_tokens = 3
    seq.draft_kv_computed_tokens = 3
    assert block_manager.ensure_capacity_for_token_position(seq, 3)
    assert seq.block_table == reserved_ids

    # Later use of the previously reserved position reuses the same physical ID.
    assert block_manager.ensure_capacity_for_token_position(seq, 8)
    assert seq.block_table == reserved_ids


def test_vectorized_acceptance_mask_and_residual_correction():
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    draft = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # Each proposal has q(x)=1 and p(x)=0, so rejection is certain.
    accepted = speculative_acceptance_mask([0, 0], target, draft)
    assert accepted == [False, False]
    assert sample_residual_correction(target[0], draft[0]) == 1
```

## 7. Step 18：安装依赖并运行单元测试

在仓库根目录执行：

```bash
python -m pip install -e ".[ui,dev]"
python -m compileall -q tinyinfer examples/advance07_speculative_decoding.py tests/test_speculative_decoding.py
node --check tinyinfer/ui/static/app.js
PYTHONPATH=. pytest -q tests/test_speculative_decoding.py
```

如果仓库已按原先方式安装过，也可以直接运行测试。测试不需要下载 Qwen3 权重，但 Python 环境需要安装项目声明的 `torch`、`transformers`、`safetensors`、`xxhash` 和 `pytest`。

### 7.1 单元测试的边界

- `test_distribution_from_logits_applies_temperature_and_top_k` 检查概率分布只保留 top-k 候选且总和为 1。
- `test_speculative_accepts_when_distributions_match` 检查 p=q 时 proposal 必然被接受。
- `test_speculative_rejection_samples_from_positive_residual` 构造一个 residual 只在 token 1 上有概率质量的例子，验证拒绝后 correction 必须落在该支持集。
- `test_rollback_clears_each_cache_to_its_own_written_frontier` 验证 Target / Draft 写入长度不同时，分别清理各自尾部且不触碰已提交 KV。
- `test_prefix_cache_registers_only_blocks_complete_in_both_models` 验证 Target 写满但 Draft 未写满的 block 不会被注册。
- `test_reserved_block_ids_survive_speculative_rollback` 验证 rollback 不缩短 `block_table`，后续写入复用同一 reservation。

这些测试验证的是可单独检验的算法与缓存 helper，不等于已经在当前会话环境运行了 Qwen3-4B 的 GPU 集成测试。本讲源码的静态语法检查通过；实际模型加载、GPU 显存预算和端到端延迟必须在你自己的 WSL + RTX 5090 环境确认。

## 8. Step 19：启动 baseline 模式

在仓库根目录执行：

```bash
python examples/advance07_speculative_decoding.py \
  --model /home/jason/huggingface/Qwen3-0.6B \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-size 16 \
  --gpu-memory-utilization 0.80
```

打开 `http://127.0.0.1:8000/`。这条命令没有 `--speculative`，因此只加载 Target-only 的 Qwen3-0.6B，右侧 SD 指标和 draft 控件隐藏。原始 `examples/advance02_ui_plus.py` 也仍然保留。

## 9. Step 20：启动 Speculative Decoding 模式

```bash
python examples/advance07_speculative_decoding.py \
  --speculative \
  --model /home/jason/huggingface/Qwen3-4B \
  --draft-model /home/jason/huggingface/Qwen3-0.6B \
  --max-draft-tokens 5 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --block-size 16 \
  --gpu-memory-utilization 0.80
```

如果显存预算不足，先按以下顺序调整，而不是直接把 `gpu_memory_utilization` 调到 1.0：

1. 降低 `--max-num-batched-tokens`，减小 dummy prefill 规模和临时激活预算。
2. 降低 `--max-model-len`，避免每条请求追求过长上下文。
3. 降低 `--gpu-memory-utilization` 前，先确认它在当前实现中代表 Engine 可使用的显存比例；调低它会进一步减少 KV block 数。
4. 把 `--max-draft-tokens` 调小。候选长度影响每轮需要额外预留的 block 数和验证临时张量。

启动后，`Performance statistics -> Cache Hit Rate` 下方会出现 `Speculative Decoding` 子栏，横向显示 `Max Draft Tokens`、从引擎启动后累计的 `Acceptance Rate` 和 `Avg Acceptance Length`。右侧 Controller 的 `Start all ready` 右侧会出现全局 `Max Draft Tokens` slider，允许范围 2～10。只要任意 User 处于 Running，slider 会被禁用；后端也会拒绝运行中的修改，因此新的 K 只会在所有 User 都 Idle 时提交，并影响之后的 SD round。

## 10. Step 21：如何验证这不是“只会跑”的实现

建议按以下顺序做 smoke test：

1. **Baseline 回归**：不传 `--speculative`，验证原来的 5 个 User panel、greedy/temperature、max tokens、TTFT/TPOT 和 KV 方格仍然工作。
2. **SD Greedy**：开启 SD，逐个用户选择 greedy，固定相同 prompt 和 max tokens。Target-only greedy 与 SD greedy 应在数学语义上对应；若出现极少数 token 差异，保存差异位置的 Target logits，检查单 token decode 与多 token verify 的数值差异，不要先归咎于接受概率公式。
3. **SD Sampling**：关闭 greedy，测试不同 temperature。输出不必与 baseline 在相同 seed 下逐 token 一致；应该检查无异常 token、拒绝路径能运行、输出可结束，并对大量独立样本做分布统计。
4. **Reject path**：在所有 User 都 Idle 时临时把 `max_draft_tokens` 调高，再选择 Draft/Target 更容易分歧的输入，观察全局 `Acceptance Rate` 与 `Avg Acceptance Length` 的变化。单靠某个 prompt 不一定保证拒绝，所以不要把“某条 prompt 没触发拒绝”当作 bug。
5. **EOS 和 max tokens**：设置很小的 max tokens，检查不会超过限制；检查 EOS 被接受或作为 correction 产生时，后续候选不会被追加到最终输出。
6. **Prefix Cache**：重复提交相同长 prompt，确认命中的是 paired-complete logical block；不再发生 Draft 命中后重建前缀。可在测试中验证 Target frontier 领先 Draft frontier 时，未完成的 block 不会注册。
7. **五用户并发**：使用 Start all ready 同时提交多个请求，确认没有 `block_table does not cover context_len`、rollback 越界或 tokenizer 不匹配错误。
8. **资源账本**：检查 `Weights` 显示 Target + Draft；一个逻辑 block 的成本显示 Target + Draft 两部分；`num_blocks` 与 SD 双 cache 预算一致。
9. **动态配置**：任意 User Running 时确认 slider 为 disabled，并验证后端直接发送 `configure_sd` 也会被拒绝；所有 User Idle 后再调整 2～10 范围内的全局 Max Draft Tokens，确认三个 SD 全局指标继续累计而不会被清零。

## 11. 常见问题与定位方式

### 11.1 `target and draft tokenizers do not share the same token-to-ID mapping`

不要只比较 tokenizer 名称或词表大小。检查是否误把其他 Qwen 版本、经过不同 tokenizer patch 的 checkpoint 或自定义词表混用。SD 的 proposal token ID 必须能被 Target 按相同 ID 解释。

### 11.2 `no memory left for paired target/draft KV blocks`

这不是采样算法错误，而是双模型权重、临时峰值和双 cache block 成本之后，预算不足以容纳一个完整 paired block。减小 `max_num_batched_tokens`、`gpu_memory_utilization` 或模型大小，检查是否有其他 GPU 进程占用显存。

### 11.3 `SD invariant violated: decode must have exactly one uncomputed pending token`

检查是否有代码绕过 `postprocess_speculative()`，直接追加了多个 token，却没有同步设置 `num_computed_tokens`；或者是否仍然对 SD decode 调用了原始 `Scheduler.postprocess()`。本讲的 SD 分支必须成对使用 `run_speculative()` 与 `postprocess_speculative()`。

### 11.4 `rollback position has no allocated physical block`

说明某模型实际写入的位置超出了共享 `block_table` 的 reservation。检查是否在任何 Draft / Target forward 前，按 `max(target_last_position, draft_last_position)` 调用 `ensure_capacity_for_token_position()`；Target verify 的 `[pending, d1, ..., dK]` 最后位置是 `start + K`。rollback 不应缩短 `block_table`。

### 11.5 UI acceptance rate 很低

低接受率不代表算法不正确，但可能没有性能收益。检查 Draft 是否与 Target 属于相近模型族、两者 tokenizer 是否一致、temperature/top-k 是否相同，以及当前任务是否适合 Draft。更大的 K 不一定更快，因为 Draft 生成 K 个 token 本身有成本，Target 验证长序列也有成本。

### 11.6 SD 启用后 TPOT p50 显示很小

本项目在一次 SD round 中可能一次性追加多个 token；前端会在该轮结束后收到完整 completion prefix。`output_token_times` 是用户可观察的输出更新时间，而不是每个内部 accepted token 的真实硬件生成时间。因此 SD 模式下不要单独用相邻 token 的 p50/p95 来断言 kernel 速度；结合 E2E、请求级 TPOT、总输出 token 数、verification rounds 和 accepted tokens 判断。若要做严谨 benchmark，应在专门的 benchmark 脚本中记录每轮 Target/Draft forward 的 CUDA event 时间，并与相同采样设置的 Target-only baseline 比较。

### 11.7 Greedy SD 与 Target-only 有少量差异

先确认 greedy 分支确实执行的是“Draft argmax 与 Target argmax 相等则接受，否则用 Target argmax 修正”，而不是错误地把非 greedy residual 公式套在 delta distribution 上。之后检查 Target-only q_len=1 和 SD verify q_len>1 是否使用了不同 attention backend / reduction shape，以及差异是否只出现在 logits 接近并列的位置。论文的 exactness 结论是分布/算法层面的，不意味着所有低精度 kernel 实现都逐 token bitwise 相同。

## 12. 工业实现里还会被继续追问的技术点

### 12.1 为什么 SD 不一定加速

SD 的收益来自把 Target 的串行 decode 调用换成较少次数的批量验证。但每轮还要执行 Draft 的 K 次自回归 forward，并且验证段越长，Target prefill-like compute、KV 写入和临时 logits 的成本越高。Draft acceptance rate、Target memory bandwidth、模型规模、batch size、attention backend 与 K 一起决定实际收益。

### 12.2 Dynamic gamma 与本讲 adaptive draft 的差别

本讲的 adaptive draft 是 acceptance EMA 的简化示范，只按接受率增减有效 K。生产实现通常会根据观测到的接受长度分布、Draft 与 Target 的每轮耗时、不同 batch/序列长度下的收益，动态选择 gamma；高接受率不一定意味着应无限增加 K，因为 Draft 额外前向也会变贵。

### 12.3 为什么工业实现通常不会把无效 KV 全部清零

KV Cache 是固定大小的 buffer，逻辑有效长度决定 attention 能读取的范围。只要 `context_len` / block table / computed frontier 正确，超出 committed prefix 的物理字节可以留在原处，并在未来覆盖时重写。本讲选择清零拒绝后缀，是为了把“物理上已经写入、逻辑上已失效”明确化，方便面试讲解与调试；在高性能版本中，可以增加 debug 开关，默认只回滚逻辑 frontier，避免额外清零开销。

### 12.4 Tree-based speculative decoding

本讲每个 Draft 只产生一条线性候选链。Tree-based SD 会同时扩展多个候选分支，Target 需要用树形 attention mask 或专用验证 kernel 计算分支 logits，并根据验证结果提交某条路径。此时 rollback 不再只是线性 token 后缀，通常还需要管理分支对应的 KV slot、共享前缀和 copy-on-write。

### 12.5 Prefix Cache 与 SD 的兼容性

本讲通过注册约束保证：Prefix Cache 只包含 Target 与 Draft 都已写满的 paired logical block，因此一次 hash 命中就能复用两份 storage，不需要两套 hash table。若运行时替换模型权重、tokenizer、RoPE / cache dtype 或 adapter，必须使旧 Prefix Cache 失效；同一 token hash 不能跨不兼容模型配置复用 KV。

### 12.6 采样处理必须完整纳入分布

top-k/top-p、temperature、repetition penalty、bad words、grammar constraint、speculative token processor 的执行顺序都可能改变 p 和 q。精确 rejection sampling 应使用真正的 Target 分布 p 和 Draft proposal 分布 q；如果 Draft 和 Target 使用不一致的约束，必须明确其数学语义，不能直接套公式。

### 12.7 其他工程细节

- EOS、停止词、最大输出 token 数和最大上下文长度必须在一次 SD round 的中途生效。
- Target verify 的每个 query row 代表“在对应前缀之后预测下一个 token”的分布；候选索引和 logits 行索引错一位，是最常见的实现 bug 之一。
- 拒绝后 correction token 是输出的一部分，但它的 KV 尚未写入；bonus token 也是输出的一部分，但其 KV 同样尚未写入。
- 两模型共享逻辑 block ID 只是一种 metadata 设计，不意味着两个 storage 可以共享，也不意味着二者每 block 字节数相同。
- 对性能报告，分开统计 proposal 数、accepted 数、每轮输出 token 数、Target verify time、Draft time 和用户可观察的 E2E/TPOT；单独报告 acceptance rate 不足以证明加速。
- SD sampling 的理论性质是输出分布与 Target 一致，而不是固定随机 seed 后每个 token 都与 Target-only 路径相同。

## 13. 面试时建议按这个顺序回答

被问到“你怎么实现 SD”时，建议按以下逻辑展开，而不是一上来就背公式：

1. **模型角色**：0.6B Draft 自回归提出 K 个 token；4B Target 对 `[pending, proposals...]` 做一次因果 forward，得到 K+1 行 next-token logits。
2. **接受规则**：非 greedy 对每个 proposal 用 `min(1,p(x)/q(x))` 接受；拒绝时按归一化的 `max(p-q,0)` 采 correction；全接受时从最后一行 Target logits 采 bonus。
3. **cache 状态**：Target 和 Draft 各有独立 KV storage，但共享一个逻辑 block ID、一套 token hash 和一张 block_table；只有双模型都写满的 block 才注册 Prefix Cache，命中后同时复用两份 KV。
4. **回滚公式**：`keep_tokens = start + 1 + accepted_count`；Target 的 `verified_end = start + K + 1`，Draft 另记 `draft_written_end`；分别清理各自写过的无效尾部，把两份 frontier 对齐到 `keep_tokens`，但保留 `block_table` 中已预留的 block ID。
5. **调度**：prefill 仍走现有 scheduler；decode items 先批量 Draft proposal，再按 draft depth batch forward，最后把每个用户的验证段放进一个 ragged Target batch。
6. **内存**：block 数按两个模型的权重与临时峰值扣除后，再除以 Target+Draft 的 paired block cost。
7. **性能判断**：接受率只是其中一个因素，最终要对比 Target-only 的端到端延迟、请求级 TPOT、目标模型调用次数与总 token throughput。

如果能区分 `seq.num_computed_tokens`（共同 committed frontier）、`target_kv_computed_tokens`、`draft_kv_computed_tokens`（两份物理 frontier），并解释为什么 rollback 只清理/回退内容而不释放预留 block ID，你就掌握了这部分最容易被追问的工程细节。

## 14. 参考资料

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023: <https://proceedings.mlr.press/v202/leviathan23a.html>
- Official Qwen3-4B model card: <https://huggingface.co/Qwen/Qwen3-4B>
- Official Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Hugging Face Hub CLI `hf download` reference: <https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cli>

---

**交付说明：** 本讲义列出的文件均为完整文件版本。请按 Step 1 到 Step 21 的顺序操作；先跑 helper、paired Prefix Cache 与 reservation 单元测试，再启动 baseline 回归，最后启动 SD。不要跳过 tokenizer 对齐、双 cache 显存估算或 rollback 验证。
