# tinyInfer Advance 05：接入 Paged Flash-Decoding

这一讲直接基于当前 `tinyInfer-master(9)` 与 Advance 04 的自定义 FlashAttention 继续修改。

上一讲已经完成：

```text
prefill / chunked prefill / decode
        ↓
gather_sequence_kv()
        ↓
连续 K/V
        ↓
自定义 FlashAttention CUDA kernel
```

这条路径在 prefill 阶段是合理的，但在 decode 阶段仍然存在一个明显问题：

```text
Paged KV Cache
      ↓
gather_sequence_kv()
      ↓
重新 materialize 连续历史 K/V
      ↓
FlashAttention
```

对于 `q_len = 1` 的 autoregressive decode，本轮真正需要的是：

```text
Q_current
    ↓
直接根据 block table 访问 paged K/V
    ↓
完成 q @ K^T、online softmax、P @ V
```

因此本讲增加一条专门针对 decode 的 **Paged Flash-Decoding** 路径。

本讲只做到：

1. prefill / chunked prefill 继续使用 Advance 04 的 FlashAttention，不改变其行为。
2. 仅当 `q_len == 1` 且 backend 为 `flash` 时进入 Flash-Decoding。
3. Flash-Decoding 直接读取 `PagedKVCache`，不再调用 `gather_sequence_kv()`。
4. 一个 Stage-1 CTA 负责一个 `q_head × logical KV block`。
5. `blockDim.x = 128`，固定 4 个 warp；每个 warp 轮流处理 KV block 内的 token。
6. 一个 warp 的 32 lanes 协作完成一个 token 的 `q·k`，因此 K/V 的 `head_dim` 方向天然 coalesced。
7. Stage 1 对每个 KV block 生成局部 `(m, l, O)`；Stage 2 再跨所有 KV blocks 合并。
8. 原始 gather + attention decode 路径不删除，在 `_attend_one()` 中整段注释保留，方便后续做 A/B 性能对比。
9. 正确处理最后一个未填满的 KV cache block。
10. 增加独立 correctness test，对比 PyTorch SDPA，并显式测试 paged physical-block 重排与 tail block。

本讲所有需要新增或修改的文件都给出**完整内容**。没有出现在本讲里的文件保持当前版本不变。

---

# 0. 最终目录变化

本讲完成后：

```text
tinyInfer/
├── setup.py                                  # [MOD] 编译 flash_decoding.cu
├── tinyinfer/
│   ├── csrc/
│   │   ├── flash_attention.cu                # 保持不变
│   │   ├── flash_attention_bind.cpp          # [MOD] 增加 Flash-Decoding binding
│   │   └── flash_decoding.cu                 # [MOD] 完成两阶段 kernel
│   └── layers/
│       ├── attention.py                      # [MOD] q_len=1 走 paged decode
│       ├── flash_attention.py                # 保持不变
│       └── flash_decoding.py                 # [NEW] Python 适配层
├── tests/
│   ├── test_flash_attention.py               # 保持不变
│   └── test_flash_decoding.py                # [NEW] paged decode correctness
└── docs/
    └── advance05_flashdecoding.md
```

最终 attention 路径变成：

```text
Attention.forward
        ↓
store_kv()                      # 本轮新 K/V 先写入 paged cache
        ↓
逐 seq 处理
        │
        ├── q_len > 1
        │      ↓
        │  gather_sequence_kv()
        │      ↓
        │  _attend_one()
        │      ↓
        │  torch SDPA / FlashAttention
        │
        └── q_len == 1 && backend == flash
               ↓
          _attend_decode_one()
               ↓
          flash_decoding.py
               ↓
          flash_attention_bind.cpp
               ↓
          flash_decoding.cu
               │
               ├── Stage 1: head × KV block
               │
               └── Stage 2: merge partial states
```

注意这一讲**不修改**：

```text
Scheduler
BlockManager
ModelRunner
Context
Qwen3
PagedKVCache storage layout
```

当前已有信息已经足够支持 decode kernel：

```text
context_len
block_table
cache_k/cache_v
block_size
```

---

# 1. Step 1：先明确 decode 阶段的数据语义

当前 `Attention.forward()` 的顺序是：

```text
Q/K/V projection
      ↓
store_kv()
      ↓
attention
```

也就是说，在真正执行 attention 之前：

```text
当前 decode token 的 K_current / V_current
```

已经写入 `PagedKVCache`。

因此本讲中的：

```text
context_len
```

统一定义为：

> 当前 query 能看到的完整 K/V token 数量，**包含当前 decode token 自己**。

例如：

```text
历史已有 100 token
当前正在 decode token #100
```

`store_kv()` 之后：

```text
context_len = 101
```

当前 query 应看到：

```text
K/V token 0 ... 100
```

因此 Flash-Decoding 不再需要 causal mask：

```text
q_len = 1
query 位于当前 context 最末尾
所有 context K/V 都合法可见
```

这也是 decode-only kernel 可以删除：

```text
query_start_pos
is_causal
mask matrix
```

的原因。

---

# 2. Step 2：为什么这一版需要两个 kernel

Advance 04 的 FlashAttention 中，一个 CTA 固定负责一个 Q tile：

```text
CTA(Q tile)
    ↓
for every K/V tile:
    QK
    online softmax
    update O
    ↓
final O
```

所以：

```text
m / l / O
```

始终留在同一个 CTA 内，可以一直放在 shared memory / register 中。

本讲为了增加 decode 阶段的 grid-level parallelism，把 KV block 提升成：

```text
grid.y
```

即：

```text
CTA(head0, KV block0)
CTA(head0, KV block1)
CTA(head0, KV block2)
...
```

不同 CTA 之间不能用：

```cpp
__syncthreads();
```

同步，所以每个 CTA 必须先输出自己的局部 softmax state：

```text
(m_block, l_block, O_block)
```

然后第二个 kernel 再合并。

本讲使用的数学状态定义为：

```text
m = 当前 partition 内最大 score
l = Σ exp(score - m)
O = Σ exp(score - m) * V
```

注意这里：

```text
O 还没有除以 l
```

只有最后 Stage 2 完成全局合并后才：

```text
O_final = O / l
```

两个状态：

```text
(m1, l1, O1)
(m2, l2, O2)
```

可以精确合并：

```text
m = max(m1, m2)

l = exp(m1-m) * l1
  + exp(m2-m) * l2

O = exp(m1-m) * O1
  + exp(m2-m) * O2
```

因此 Stage 1 如何切 KV 都不会改变最终 attention 数值。

---

# 3. Step 3：固定 Stage-1 并行策略

本讲暂时固定：

```text
grid.x = q_heads
grid.y = num_kvcache_block

block.x = 128
        = 4 warps
```

其中：

```text
blockIdx.x -> q_head
blockIdx.y -> logical KV block
```

因此一个 CTA 负责：

```text
一个 q_head
×
一个 logical KV block 中的所有有效 token
```

## 3.1 warp 如何处理 token

4 个 warp：

```text
warp 0
warp 1
warp 2
warp 3
```

以步长 4 轮流处理 token。

当：

```text
kvcache_block_size = 16
```

时：

```text
warp0 -> token 0, 4, 8, 12
warp1 -> token 1, 5, 9, 13
warp2 -> token 2, 6, 10, 14
warp3 -> token 3, 7, 11, 15
```

kernel 内对应：

```cpp
for (
    int token_off = warp_id;
    token_off < valid_tokens;
    token_off += 4
) {
    ...
}
```

## 3.2 为什么不是一个 thread 对应一个 token

假设：

```text
head_dim = 128
```

如果一个 thread 独自处理一个 token，那么它需要独立计算：

```text
128 维 dot product
```

而且同一个 warp 中不同 lane 会跨 token stride 读取 K，访存不理想。

本讲采用：

```text
一个 warp -> 一个 token
```

32 lanes 协作：

```text
lane0  -> d = 0, 32, 64, 96
lane1  -> d = 1, 33, 65, 97
...
lane31 -> d = 31, 63, 95, 127
```

因此固定 token 后：

```text
K[token, kv_head, 0:32]
```

由整个 warp 连续读取，更利于 coalesced global-memory access。

---

# 4. Step 4：完成 `tinyinfer/csrc/flash_decoding.cu`

修改：

```text
tinyinfer/csrc/flash_decoding.cu
```

这一版包含：

```text
1. dtype conversion helper
2. warp reduction helper
3. Stage-1 kernel_flash_decoding
4. Stage-2 kernel_flash_decoding_reduce
5. typed launcher
6. extern "C" launcher
```

注意中间状态统一使用：

```text
float
```

而不是输入 dtype `T`。

原因是：

```text
BF16/FP16 输入
```

可以接受，但：

```text
softmax max
softmax denominator
partial output accumulator
```

不应该继续用 BF16/FP16 保存。

完整文件如下：

```cpp
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>


#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err));                   \
        }                                                                       \
    } while (0)


// [MOD] Flash-Decoding 固定一个 CTA = 4 warps = 128 threads。
constexpr int FLASH_DECODING_NUM_WARPS = 4;
constexpr int FLASH_DECODING_THREADS = 128;
constexpr unsigned FLASH_DECODING_MASK = 0xffffffffu;


template <typename T>
__device__ inline float to_float_decoding(T x) {
    return static_cast<float>(x);
}


template <>
__device__ inline float to_float_decoding<half>(half x) {
    return __half2float(x);
}


template <>
__device__ inline float to_float_decoding<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}


template <typename T>
__device__ inline T from_float_decoding(float x) {
    return static_cast<T>(x);
}


template <>
__device__ inline half from_float_decoding<half>(float x) {
    return __float2half(x);
}


template <>
__device__ inline __nv_bfloat16 from_float_decoding<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}


// [MOD] 一个 warp 协作完成一个 token 的 q·k。
__device__ __forceinline__ float warp_reduce_sum_decoding(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FLASH_DECODING_MASK, val, offset);
    }
    return __shfl_sync(FLASH_DECODING_MASK, val, 0);
}


// [MOD] Stage 1: 一个 CTA 处理一个 q_head × 一个 logical KV block。
template <typename T>
__global__ void kernel_flash_decoding(
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    float scale,

    const T* __restrict__ q,
    const T* __restrict__ KCache,
    const T* __restrict__ VCache,
    const int* __restrict__ table_list,
    int num_kvcache_block,
    int kvcache_block_size,

    float* __restrict__ partial_l,
    float* __restrict__ partial_m,
    float* __restrict__ partial_O
) {
    const int q_head = blockIdx.x;
    const int logical_block = blockIdx.y;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (q_head >= q_heads || logical_block >= num_kvcache_block) {
        return;
    }

    // [MOD] 本讲固定 block.x=128，host launcher 同样固定该值。
    if (blockDim.x != FLASH_DECODING_THREADS) {
        return;
    }

    // [MOD] GQA 映射: 多个 Q heads 共享一个 KV head。
    const int q_per_kv = q_heads / kv_heads;
    const int kv_head = q_head / q_per_kv;

    // [MOD] 当前 logical KV block 对应的全局 token 起点。
    const int token_start = logical_block * kvcache_block_size;

    if (token_start >= context_len) {
        return;
    }

    // [MOD] tail block 可能没有填满，只处理真实 token。
    const int valid_tokens = min(
        kvcache_block_size,
        context_len - token_start
    );

    // [MOD] PagedAttention 核心: logical -> physical block。
    const int physical_block = table_list[logical_block];

    // 正常情况下真实 logical block 不允许为 -1。
    if (physical_block < 0) {
        return;
    }

    // shared memory layout:
    // q_sm[head_dim]
    // warp_O[4][head_dim]
    // warp_m[4]
    // warp_l[4]
    // block_m[1]
    // block_l[1]
    extern __shared__ float shared_mem[];

    float* q_sm = shared_mem;
    float* warp_O = q_sm + head_dim;
    float* warp_m = warp_O + FLASH_DECODING_NUM_WARPS * head_dim;
    float* warp_l = warp_m + FLASH_DECODING_NUM_WARPS;
    float* block_m = warp_l + FLASH_DECODING_NUM_WARPS;
    float* block_l = block_m + 1;

    // [MOD] Step 1: Q 只有一个 token，整个 CTA 协作缓存当前 q_head。
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_sm[d] = to_float_decoding<T>(
            q[q_head * head_dim + d]
        );
    }

    // [MOD] 每个 warp 独立维护一个 partial output state。
    for (int d = lane_id; d < head_dim; d += 32) {
        warp_O[warp_id * head_dim + d] = 0.0f;
    }

    if (lane_id == 0) {
        warp_m[warp_id] = -CUDART_INF_F;
        warp_l[warp_id] = 0.0f;
    }

    __syncthreads();

    float local_m = -CUDART_INF_F;
    float local_l = 0.0f;

    // [MOD] Step 2: 4 个 warp 以步长 4 轮流处理 token。
    for (
        int token_off = warp_id;
        token_off < valid_tokens;
        token_off += FLASH_DECODING_NUM_WARPS
    ) {
        // K/V layout:
        // [physical_block, token_off, kv_head, head_dim]
        const size_t kv_base =
            (
                (
                    static_cast<size_t>(physical_block)
                    * static_cast<size_t>(kvcache_block_size)
                    + static_cast<size_t>(token_off)
                )
                * static_cast<size_t>(kv_heads)
                + static_cast<size_t>(kv_head)
            )
            * static_cast<size_t>(head_dim);

        // [MOD] Step 2.1: 一个 warp 协作完成 q·k。
        float dot = 0.0f;

        for (int d = lane_id; d < head_dim; d += 32) {
            const float q_val = q_sm[d];
            const float k_val = to_float_decoding<T>(
                KCache[kv_base + static_cast<size_t>(d)]
            );
            dot += q_val * k_val;
        }

        dot = warp_reduce_sum_decoding(dot);
        const float score = dot * scale;

        // [MOD] Step 2.2: warp-local online softmax。
        const float new_m = fmaxf(local_m, score);

        const float alpha =
            (local_l == 0.0f)
                ? 0.0f
                : expf(local_m - new_m);

        const float beta = expf(score - new_m);

        // [MOD] Step 2.3: O <- alpha * O + beta * V。
        for (int d = lane_id; d < head_dim; d += 32) {
            const float v_val = to_float_decoding<T>(
                VCache[kv_base + static_cast<size_t>(d)]
            );

            float& out = warp_O[warp_id * head_dim + d];
            out = alpha * out + beta * v_val;
        }

        local_l = alpha * local_l + beta;
        local_m = new_m;
    }

    // [MOD] 空闲 warp 在 tail block 中保持 l=0，后续 merge 自动忽略。
    if (lane_id == 0) {
        warp_m[warp_id] = local_m;
        warp_l[warp_id] = local_l;
    }

    __syncthreads();

    // [MOD] Step 3: CTA 内先合并 4 个 warp 的 m/l。
    if (tid == 0) {
        float merged_m = -CUDART_INF_F;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_m = fmaxf(merged_m, warp_m[w]);
            }
        }

        float merged_l = 0.0f;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_l +=
                    expf(warp_m[w] - merged_m)
                    * warp_l[w];
            }
        }

        block_m[0] = merged_m;
        block_l[0] = merged_l;
    }

    __syncthreads();

    const float merged_m = block_m[0];
    const float merged_l = block_l[0];

    const size_t partial_idx =
        static_cast<size_t>(q_head)
        * static_cast<size_t>(num_kvcache_block)
        + static_cast<size_t>(logical_block);

    // [MOD] Step 4: CTA 内合并 4 个 warp 的 numerator O。
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float merged_O = 0.0f;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                const float correction = expf(warp_m[w] - merged_m);
                merged_O +=
                    correction
                    * warp_O[w * head_dim + d];
            }
        }

        partial_O[
            partial_idx * static_cast<size_t>(head_dim)
            + static_cast<size_t>(d)
        ] = merged_O;
    }

    if (tid == 0) {
        partial_m[partial_idx] = merged_m;
        partial_l[partial_idx] = merged_l;
    }
}


// [MOD] Stage 2: 一个 CTA 负责一个 q_head，合并所有 KV-block partial states。
template <typename T>
__global__ void kernel_flash_decoding_reduce(
    int q_heads,
    int head_dim,
    int num_kvcache_block,

    const float* __restrict__ partial_l,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_O,

    T* __restrict__ O
) {
    const int q_head = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (q_head >= q_heads) {
        return;
    }

    if (blockDim.x != FLASH_DECODING_THREADS) {
        return;
    }

    // shared memory layout:
    // warp_O[4][head_dim]
    // warp_m[4]
    // warp_l[4]
    // final_m[1]
    // final_l[1]
    extern __shared__ float shared_mem[];

    float* warp_O = shared_mem;
    float* warp_m = warp_O + FLASH_DECODING_NUM_WARPS * head_dim;
    float* warp_l = warp_m + FLASH_DECODING_NUM_WARPS;
    float* final_m = warp_l + FLASH_DECODING_NUM_WARPS;
    float* final_l = final_m + 1;

    for (int d = lane_id; d < head_dim; d += 32) {
        warp_O[warp_id * head_dim + d] = 0.0f;
    }

    if (lane_id == 0) {
        warp_m[warp_id] = -CUDART_INF_F;
        warp_l[warp_id] = 0.0f;
    }

    __syncthreads();

    float local_m = -CUDART_INF_F;
    float local_l = 0.0f;

    // [MOD] 4 个 warp 同样以步长 4 轮流处理 Stage-1 partial blocks。
    for (
        int block_id = warp_id;
        block_id < num_kvcache_block;
        block_id += FLASH_DECODING_NUM_WARPS
    ) {
        const size_t partial_idx =
            static_cast<size_t>(q_head)
            * static_cast<size_t>(num_kvcache_block)
            + static_cast<size_t>(block_id);

        const float block_l_value = partial_l[partial_idx];

        if (block_l_value <= 0.0f) {
            continue;
        }

        const float block_m_value = partial_m[partial_idx];
        const float new_m = fmaxf(local_m, block_m_value);

        const float alpha =
            (local_l == 0.0f)
                ? 0.0f
                : expf(local_m - new_m);

        const float beta = expf(block_m_value - new_m);

        // partial_O 本身是以 block_m 为基准的 numerator，不能提前 / block_l。
        for (int d = lane_id; d < head_dim; d += 32) {
            const float block_O = partial_O[
                partial_idx * static_cast<size_t>(head_dim)
                + static_cast<size_t>(d)
            ];

            float& out = warp_O[warp_id * head_dim + d];
            out = alpha * out + beta * block_O;
        }

        local_l =
            alpha * local_l
            + beta * block_l_value;

        local_m = new_m;
    }

    if (lane_id == 0) {
        warp_m[warp_id] = local_m;
        warp_l[warp_id] = local_l;
    }

    __syncthreads();

    // [MOD] 先合并4个warp的全局 m/l。
    if (tid == 0) {
        float merged_m = -CUDART_INF_F;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_m = fmaxf(merged_m, warp_m[w]);
            }
        }

        float merged_l = 0.0f;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_l +=
                    expf(warp_m[w] - merged_m)
                    * warp_l[w];
            }
        }

        final_m[0] = merged_m;
        final_l[0] = merged_l;
    }

    __syncthreads();

    const float merged_m = final_m[0];
    const float merged_l = final_l[0];

    // [MOD] 最后合并 numerator，并且只在这里执行 O /= l。
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float merged_O = 0.0f;

#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                const float correction = expf(warp_m[w] - merged_m);
                merged_O +=
                    correction
                    * warp_O[w * head_dim + d];
            }
        }

        O[q_head * head_dim + d] =
            from_float_decoding<T>(merged_O / merged_l);
    }
}


// [MOD] dtype-specialized launcher。
template <typename T>
void launch_flash_decoding_typed(
    const T* q,
    const T* k_cache,
    const T* v_cache,
    const int* block_table,
    T* out,
    float* partial_l,
    float* partial_m,
    float* partial_O,
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int num_kvcache_block,
    int kvcache_block_size,
    float scale
) {
    const dim3 stage1_grid(q_heads, num_kvcache_block);
    const dim3 stage1_block(FLASH_DECODING_THREADS);

    const size_t stage1_smem_size =
        (
            head_dim
            + FLASH_DECODING_NUM_WARPS * head_dim
            + FLASH_DECODING_NUM_WARPS
            + FLASH_DECODING_NUM_WARPS
            + 2
        )
        * sizeof(float);

    kernel_flash_decoding<T><<<
        stage1_grid,
        stage1_block,
        stage1_smem_size
    >>>(
        context_len,
        q_heads,
        kv_heads,
        head_dim,
        scale,
        q,
        k_cache,
        v_cache,
        block_table,
        num_kvcache_block,
        kvcache_block_size,
        partial_l,
        partial_m,
        partial_O
    );

    CUDA_CHECK(cudaGetLastError());

    const dim3 stage2_grid(q_heads);
    const dim3 stage2_block(FLASH_DECODING_THREADS);

    const size_t stage2_smem_size =
        (
            FLASH_DECODING_NUM_WARPS * head_dim
            + FLASH_DECODING_NUM_WARPS
            + FLASH_DECODING_NUM_WARPS
            + 2
        )
        * sizeof(float);

    kernel_flash_decoding_reduce<T><<<
        stage2_grid,
        stage2_block,
        stage2_smem_size
    >>>(
        q_heads,
        head_dim,
        num_kvcache_block,
        partial_l,
        partial_m,
        partial_O,
        out
    );

    CUDA_CHECK(cudaGetLastError());
}


// [MOD] C++ binding 只需要传裸 device pointer 与 runtime metadata。
extern "C" void flash_decoding_forward_cuda(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int* block_table,
    void* out,
    float* partial_l,
    float* partial_m,
    float* partial_O,
    int dtype_code,
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int num_kvcache_block,
    int kvcache_block_size,
    float scale
) {
    if (dtype_code == 0) {
        launch_flash_decoding_typed<half>(
            static_cast<const half*>(q),
            static_cast<const half*>(k_cache),
            static_cast<const half*>(v_cache),
            block_table,
            static_cast<half*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }

    if (dtype_code == 1) {
        launch_flash_decoding_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k_cache),
            static_cast<const __nv_bfloat16*>(v_cache),
            block_table,
            static_cast<__nv_bfloat16*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }

    if (dtype_code == 2) {
        launch_flash_decoding_typed<float>(
            static_cast<const float*>(q),
            static_cast<const float*>(k_cache),
            static_cast<const float*>(v_cache),
            block_table,
            static_cast<float*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code");
}
```

---

# 5. Step 5：理解 tail block 为什么不需要 mask

假设：

```text
context_len = 37
kvcache_block_size = 16
```

则：

```text
num_kvcache_block = ceil(37 / 16) = 3
```

三个 logical blocks：

```text
block0 -> 16 tokens
block1 -> 16 tokens
block2 -> 5 tokens
```

Stage 1 对 block2：

```cpp
valid_tokens = min(
    16,
    37 - 2 * 16
);
```

得到：

```text
valid_tokens = 5
```

因此第一轮：

```text
warp0 -> token_off 0
warp1 -> token_off 1
warp2 -> token_off 2
warp3 -> token_off 3
```

第二轮只有：

```text
warp0 -> token_off 4
```

另外三个 warp 的：

```text
token_off >= valid_tokens
```

不会进入 loop。

所以 tail 处理完全由：

```cpp
token_off < valid_tokens
```

解决，不需要：

```text
causal mask
padding score
-inf mask matrix
```

---

# 6. Step 6：修改 C++ Binding

修改：

```text
tinyinfer/csrc/flash_attention_bind.cpp
```

这一讲继续复用同一个：

```text
tinyinfer._C
```

extension module，不额外创建第二个 `.so`。

Binding 新增：

```text
flash_decoding_forward(...)
```

它负责：

```text
1. 检查 q/cache/block_table shape。
2. 根据 context_len 计算真实 num_kvcache_block。
3. 为 Stage 1 分配 float32 partial buffers。
4. 分配最终 out。
5. 调用 flash_decoding_forward_cuda()。
```

这里特别注意：

```text
block_table.shape[0]
```

可能大于当前 sequence 实际需要的 block 数，因为 `ModelRunner` 会按 batch 最大 block-table 长度 padding `-1`。

因此真实：

```text
num_kvcache_block
```

必须计算为：

```text
ceil(context_len / kvcache_block_size)
```

而不能直接使用：

```text
block_table.numel()
```

完整文件如下：

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


// 定义于 flash_attention.cu
extern "C" void flash_attention_forward_cuda(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int dtype_code,
    int target_seq_len,
    int src_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int query_start_pos,
    float scale,
    bool is_causal
);


// [MOD] 定义于 flash_decoding.cu
extern "C" void flash_decoding_forward_cuda(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int* block_table,
    void* out,
    float* partial_l,
    float* partial_m,
    float* partial_O,
    int dtype_code,
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int num_kvcache_block,
    int kvcache_block_size,
    float scale
);


// python 端 FlashAttention 入口。
torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t query_start_pos,
    double scale,
    bool is_causal
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    TORCH_CHECK(q.device() == k.device(), "q/k must be on the same device");
    TORCH_CHECK(q.device() == v.device(), "q/v must be on the same device");

    TORCH_CHECK(q.dim() == 3, "q must have shape [Tq, Hq, D]");
    TORCH_CHECK(k.dim() == 3, "k must have shape [Tk, Hkv, D]");
    TORCH_CHECK(v.dim() == 3, "v must have shape [Tk, Hkv, D]");

    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shape mismatch");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k head_dim mismatch");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "Hq must be divisible by Hkv");

    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    const int64_t tq = q.size(0);
    const int64_t tk = k.size(0);

    TORCH_CHECK(tq > 0, "Tq must be positive");
    TORCH_CHECK(tk > 0, "Tk must be positive");
    TORCH_CHECK(query_start_pos >= 0, "query_start_pos must be non-negative");
    TORCH_CHECK(
        query_start_pos + tq <= tk,
        "query range must be covered by K/V history"
    );

    int dtype_code = -1;
    if (q.scalar_type() == at::kHalf) {
        dtype_code = 0;
    } else if (q.scalar_type() == at::kBFloat16) {
        dtype_code = 1;
    } else if (q.scalar_type() == at::kFloat) {
        dtype_code = 2;
    } else {
        TORCH_CHECK(false, "FlashAttention supports fp16, bf16 and fp32 only");
    }

    c10::cuda::CUDAGuard device_guard(q.device());
    auto out = torch::empty_like(q);

    flash_attention_forward_cuda(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        dtype_code,
        static_cast<int>(tq),
        static_cast<int>(tk),
        static_cast<int>(q.size(1)),
        static_cast<int>(k.size(1)),
        static_cast<int>(q.size(2)),
        static_cast<int>(query_start_pos),
        static_cast<float>(scale),
        is_causal
    );

    return out;
}


// [MOD] python 端 Paged Flash-Decoding 入口。
torch::Tensor flash_decoding_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    int64_t context_len,
    int64_t kvcache_block_size,
    double scale
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");

    TORCH_CHECK(q.device() == k_cache.device(), "q/k_cache device mismatch");
    TORCH_CHECK(q.device() == v_cache.device(), "q/v_cache device mismatch");
    TORCH_CHECK(q.device() == block_table.device(), "q/block_table device mismatch");

    TORCH_CHECK(q.dim() == 3, "q must have shape [1, Hq, D]");
    TORCH_CHECK(q.size(0) == 1, "Flash-Decoding only supports q_len == 1");

    TORCH_CHECK(
        k_cache.dim() == 4,
        "k_cache must have shape [num_blocks, block_size, Hkv, D]"
    );
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "k/v cache shape mismatch");

    TORCH_CHECK(block_table.dim() == 1, "block_table must be 1-D");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");

    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q/k_cache dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q/v_cache dtype mismatch");

    TORCH_CHECK(q.size(2) == k_cache.size(3), "q/cache head_dim mismatch");
    TORCH_CHECK(q.size(1) % k_cache.size(2) == 0, "Hq must be divisible by Hkv");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");

    TORCH_CHECK(context_len > 0, "context_len must be positive");
    TORCH_CHECK(kvcache_block_size > 0, "kvcache_block_size must be positive");
    TORCH_CHECK(
        k_cache.size(1) == kvcache_block_size,
        "kvcache_block_size does not match cache shape"
    );

    // [MOD] 真实 logical-block 数由 context_len 决定，不能直接使用 padded table 长度。
    const int64_t num_kvcache_block =
        (context_len + kvcache_block_size - 1)
        / kvcache_block_size;

    TORCH_CHECK(
        block_table.numel() >= num_kvcache_block,
        "block_table does not cover context_len"
    );

    int dtype_code = -1;
    if (q.scalar_type() == at::kHalf) {
        dtype_code = 0;
    } else if (q.scalar_type() == at::kBFloat16) {
        dtype_code = 1;
    } else if (q.scalar_type() == at::kFloat) {
        dtype_code = 2;
    } else {
        TORCH_CHECK(false, "Flash-Decoding supports fp16, bf16 and fp32 only");
    }

    c10::cuda::CUDAGuard device_guard(q.device());

    auto out = torch::empty_like(q);

    auto float_options = q.options().dtype(torch::kFloat32);

    // [MOD] Stage 1 中间状态全部使用 float32。
    auto partial_l = torch::empty(
        {q.size(1), num_kvcache_block},
        float_options
    );

    auto partial_m = torch::empty(
        {q.size(1), num_kvcache_block},
        float_options
    );

    auto partial_O = torch::empty(
        {q.size(1), num_kvcache_block, q.size(2)},
        float_options
    );

    flash_decoding_forward_cuda(
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        block_table.data_ptr<int>(),
        out.data_ptr(),
        partial_l.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_O.data_ptr<float>(),
        dtype_code,
        static_cast<int>(context_len),
        static_cast<int>(q.size(1)),
        static_cast<int>(k_cache.size(2)),
        static_cast<int>(q.size(2)),
        static_cast<int>(num_kvcache_block),
        static_cast<int>(kvcache_block_size),
        static_cast<float>(scale)
    );

    return out;
}


// [MOD] 同一个 tinyinfer._C 同时注册两个 CUDA backend。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_attention_forward",
        &flash_attention_forward,
        "tinyInfer FlashAttention forward",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("query_start_pos"),
        pybind11::arg("scale"),
        pybind11::arg("is_causal") = true
    );

    // [MOD]
    m.def(
        "flash_decoding_forward",
        &flash_decoding_forward,
        "tinyInfer Paged Flash-Decoding forward",
        pybind11::arg("q"),
        pybind11::arg("k_cache"),
        pybind11::arg("v_cache"),
        pybind11::arg("block_table"),
        pybind11::arg("context_len"),
        pybind11::arg("kvcache_block_size"),
        pybind11::arg("scale")
    );
}
```

---

# 7. Step 7：新增 Python Flash-Decoding 适配层

新增：

```text
tinyinfer/layers/flash_decoding.py
```

完整文件：

```python
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_extension():
    try:
        from tinyinfer import _C
    except ImportError as exc:
        raise RuntimeError(
            "tinyinfer._C is not built; run: "
            "MAX_JOBS=4 python -m pip install -e . --no-build-isolation"
        ) from exc
    return _C


# [NEW] q_len=1 专用 Paged Flash-Decoding Python wrapper。
def flash_decoding(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    kvcache_block_size: int,
    scale: float,
) -> torch.Tensor:
    if q.device.type != "cuda":
        raise ValueError("Flash-Decoding requires CUDA tensors")
    if q.ndim != 3 or q.shape[0] != 1:
        raise ValueError("Flash-Decoding requires q shape [1, Hq, D]")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "K/V cache must be [num_blocks, block_size, Hkv, D]"
        )
    if int(context_len) <= 0:
        raise ValueError("context_len must be positive")

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous()

    return _load_extension().flash_decoding_forward(
        q,
        k_cache,
        v_cache,
        block_table,
        int(context_len),
        int(kvcache_block_size),
        float(scale),
    )
```

这一层没有：

```text
gather_sequence_kv()
repeat_interleave()
```

因为：

```text
paged addressing
GQA head mapping
```

都在 CUDA kernel 内完成。

---

# 8. Step 8：修改 `setup.py`

修改仓库根目录：

```text
setup.py
```

完整文件：

```python
# MAX_JOBS=4 python -m pip install -e . --no-build-isolation 会自动调用该文件

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="tinyinfer",
    ext_modules=[
        CUDAExtension(
            name="tinyinfer._C",
            sources=[
                "tinyinfer/csrc/flash_attention_bind.cpp",
                "tinyinfer/csrc/flash_attention.cu",
                # [MOD] 增加 Paged Flash-Decoding CUDA translation unit。
                "tinyinfer/csrc/flash_decoding.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
```

重新编译：

```bash
MAX_JOBS=4 python -m pip install -e . --no-build-isolation
```

验证：

```bash
python - <<'PY'
from tinyinfer import _C

print("flash attention:", hasattr(_C, "flash_attention_forward"))
print("flash decoding:", hasattr(_C, "flash_decoding_forward"))
PY
```

预期：

```text
flash attention: True
flash decoding: True
```

---

# 9. Step 9：修改 `tinyinfer/layers/attention.py`

这一层是本讲 runtime 接入的核心。

当前路径无论 prefill 还是 decode 都会：

```text
gather_sequence_kv()
```

本讲修改为：

```text
backend == flash && q_len == 1
        ↓
直接 flash_decoding(
    q_i,
    cache_k,
    cache_v,
    block_table,
    context_len
)
```

而：

```text
q_len > 1
```

继续使用原来的：

```text
gather + _attend_one
```

这样本讲只改变 decode data path，不触碰 prefill。

## 9.1 关于保留旧实现

为了方便后续 benchmark，本讲在 `_attend_one()` 中保留一段完整的原始 attention 实现，并整体注释。

正常执行时：

```text
prefill -> 当前 active implementation
```

如果以后要把 decode 暂时退回旧实现比较性能，只需要按注释中的说明恢复：

```text
gather_sequence_kv + _attend_one
```

而不需要重新从 Git 历史寻找旧代码。

修改后的完整文件如下：

```python
import os
import torch
from torch import nn

from tinyinfer.utils.context import get_context

from tinyinfer.layers.flash_attention import flash_attention
# [MOD] decode-only paged Flash-Decoding backend。
from tinyinfer.layers.flash_decoding import flash_decoding


# 一些独立的辅助函数
def store_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    # cache_k/v : [num_blocks, block_size, num_heads, head_dim]
    # k/v: [num_tokens, num_kv_heads, head_dim], 这里的k/v可能来自很多个seq请求且每个请求长度不一;
    # 比如seq0刚完成prefill需要存4个token对应的kv cache, 而seq1/2则都是在decode阶段、只需存储1个token;
    # 则此时num_tokens = 4 + 1 + 1 = 6
    if k.shape != v.shape:
        raise ValueError("k/v shape mismatch")
    if k.shape[0] != slot_mapping.numel():
        raise ValueError("slot_mapping length must equal token count")
    for i, slot in enumerate(slot_mapping.tolist()):
        block = slot // block_size
        offset = slot % block_size
        cache_k[block, offset].copy_(k[i])
        cache_v[block, offset].copy_(v[i])


# 针对某个请求seq获取其所有context对应的K/V cache
# cache shape: [num_blocks, block_size, num_heads, head_dim]
def gather_sequence_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
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
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

        self.register_buffer(
            "storage",
            torch.empty(shape, dtype=dtype, device=device),
            persistent=False,
        )

    def layer_kv(self, layer_idx: int):
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("layer_idx out of range")
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]


_FLASH_LOGGED = False
_TORCH_LOGGED = False
_FLASH_DECODING_LOGGED = False


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

        self.backend = os.getenv(
            "TINYINFER_ATTENTION_BACKEND",
            "torch",
        ).strip().lower()

        if self.backend not in {"torch", "flash"}:
            raise ValueError(
                "TINYINFER_ATTENTION_BACKEND must be 'torch' or 'flash'"
            )

    def set_kv_cache(self, kv_cache: PagedKVCache) -> None:
        if kv_cache.block_size != self.block_size:
            raise ValueError("KV cache block_size mismatch")
        self.kv_cache = kv_cache

    def _repeat(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_repeated = k.repeat_interleave(repeat, dim=1)
            v_repeated = v.repeat_interleave(repeat, dim=1)
            return k_repeated, v_repeated

        return k, v

    # prefill/chunked-prefill统一attention计算。
    def _attend_one(self, q_i, k_hist, v_hist, query_start_pos: int):
        # q_i:    [Tq, Hq, D]
        # k_hist: [Tk, Hkv, D]

        # [MOD] active implementation: flash backend继续使用Advance04 FlashAttention。
        if self.backend == "flash":
            global _FLASH_LOGGED
            if not _FLASH_LOGGED:
                print(
                    "[tinyInfer] prefill attention backend: "
                    "self-implemented flash attention"
                )
                _FLASH_LOGGED = True

            return flash_attention(
                q_i,
                k_hist,
                v_hist,
                query_start_pos=query_start_pos,
                scale=self.scale,
                is_causal=True,
            )

        global _TORCH_LOGGED
        if not _TORCH_LOGGED:
            print(
                "[tinyInfer] attention backend: "
                "torch.scaled_dot_product_attention"
            )
            _TORCH_LOGGED = True

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

        # ------------------------------------------------------------------
        # [MOD] 保留的原始 attention 实现，不删除，供后续性能对比。
        # 如果希望把 flash decode 临时退回旧的 gather + attention 路径，
        # 可以在 forward() 的 q_len == 1 分支恢复 gather，然后调用这里。
        # ------------------------------------------------------------------
        # if self.backend == "flash":
        #     return flash_attention(
        #         q_i,
        #         k_hist,
        #         v_hist,
        #         query_start_pos=query_start_pos,
        #         scale=self.scale,
        #         is_causal=True,
        #     )
        #
        # q = q_i.transpose(0, 1).unsqueeze(0)
        # k = k_hist.transpose(0, 1).unsqueeze(0)
        # v = v_hist.transpose(0, 1).unsqueeze(0)
        #
        # k, v = self._repeat(k, v)
        #
        # tq = q_i.shape[0]
        # tk = k_hist.shape[0]
        #
        # q_pos = torch.arange(
        #     query_start_pos,
        #     query_start_pos + tq,
        #     device=q.device,
        # )
        # k_pos = torch.arange(tk, device=q.device)
        #
        # causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        # causal = causal.unsqueeze(0).unsqueeze(0)
        #
        # out = torch.nn.functional.scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     attn_mask=causal,
        #     is_causal=False,
        #     scale=self.scale,
        # )
        #
        # return out.squeeze(0).transpose(0, 1)

    # [MOD] q_len=1专用，不materialize历史连续K/V。
    def _attend_decode_one(
        self,
        q_i: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        if q_i.shape[0] != 1:
            raise ValueError("_attend_decode_one requires q_len == 1")

        global _FLASH_DECODING_LOGGED
        if not _FLASH_DECODING_LOGGED:
            print(
                "[tinyInfer] decode attention backend: "
                "self-implemented paged flash decoding"
            )
            _FLASH_DECODING_LOGGED = True

        return flash_decoding(
            q=q_i,
            k_cache=cache_k,
            v_cache=cache_v,
            block_table=block_table,
            context_len=context_len,
            kvcache_block_size=self.block_size,
            scale=self.scale,
        )

    # interface
    def forward(self, q, k, v):
        # q/k/v:
        # [num_tokens, num_heads/num_kv_heads, head_dim]
        ctx = get_context()

        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx)

        # 本轮新K/V先写入paged cache，decode current token因此已经包含在cache中。
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

            # [MOD] flash backend + q_len=1直接进入Paged Flash-Decoding。
            # 不再执行 gather_sequence_kv()。
            if self.backend == "flash" and q_len == 1:
                outputs.append(
                    self._attend_decode_one(
                        q_i=q_i,
                        cache_k=cache_k,
                        cache_v=cache_v,
                        block_table=ctx.block_tables[i],
                        context_len=context_len,
                    )
                )
                continue

            # prefill/chunked prefill以及torch reference路径保持原实现。
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

## 9.2 为什么只在 `backend == "flash"` 时启用 Flash-Decoding

这样环境变量仍然可以直接做 reference 对照：

```bash
TINYINFER_ATTENTION_BACKEND=torch
```

得到：

```text
decode
  ↓
gather paged KV
  ↓
torch SDPA
```

而：

```bash
TINYINFER_ATTENTION_BACKEND=flash
```

得到：

```text
prefill
  ↓
FlashAttention

decode
  ↓
Paged Flash-Decoding
```

这比再增加：

```text
flash_decode / flash_prefill / torch_decode / ...
```

大量 backend 名称更简单。

---

# 10. Step 10：增加独立 correctness test

新增：

```text
tests/test_flash_decoding.py
```

这个测试需要覆盖三个本讲新增语义：

```text
1. physical KV blocks不是logical顺序排列。
2. GQA: Hq != Hkv。
3. 最后一个logical KV block只填一部分token。
```

例如：

```text
logical block 0 -> physical block 5
logical block 1 -> physical block 1
logical block 2 -> physical block 7
```

而：

```text
context_len = 37
block_size = 16
```

最后一个 physical block 只使用：

```text
5 tokens
```

完整文件：

```python
import pytest
import torch
import torch.nn.functional as F

from tinyinfer.layers.flash_decoding import flash_decoding


CUDA_AVAILABLE = torch.cuda.is_available()


def gather_reference(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
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


def torch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    k = gather_reference(
        k_cache,
        block_table,
        context_len,
        block_size,
    )
    v = gather_reference(
        v_cache,
        block_table,
        context_len,
        block_size,
    )

    hq = q.shape[1]
    hkv = k.shape[1]

    q_ref = q.transpose(0, 1).unsqueeze(0)
    k_ref = k.transpose(0, 1).unsqueeze(0)
    v_ref = v.transpose(0, 1).unsqueeze(0)

    if hq != hkv:
        repeat = hq // hkv
        k_ref = k_ref.repeat_interleave(repeat, dim=1)
        v_ref = v_ref.repeat_interleave(repeat, dim=1)

    # q_len=1且当前token位于context末尾，因此所有K/V均合法可见。
    out = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=None,
        is_causal=False,
        scale=scale,
    )

    return out.squeeze(0).transpose(0, 1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "block_size,context_len,physical_blocks",
    [
        (16, 37, [5, 1, 7]),
        (32, 65, [6, 2, 9]),
        (64, 129, [8, 3, 10]),
    ],
)
def test_flash_decoding_matches_torch(
    dtype,
    block_size,
    context_len,
    physical_blocks,
):
    torch.manual_seed(0)

    hq = 8
    hkv = 2
    head_dim = 128
    scale = head_dim ** -0.5

    num_physical_blocks = max(physical_blocks) + 2

    q = torch.randn(
        1,
        hq,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1

    k_cache = torch.randn(
        num_physical_blocks,
        block_size,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1

    v_cache = torch.randn(
        num_physical_blocks,
        block_size,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1

    # [MOD] 尾部额外增加-1，模拟ModelRunner中batch block-table padding。
    block_table = torch.tensor(
        physical_blocks + [-1, -1],
        device="cuda",
        dtype=torch.int32,
    )

    expected = torch_reference(
        q,
        k_cache,
        v_cache,
        block_table,
        context_len,
        block_size,
        scale,
    )

    actual = flash_decoding(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        context_len=context_len,
        kvcache_block_size=block_size,
        scale=scale,
    )

    if dtype == torch.bfloat16:
        atol = 0.10
        rtol = 0.10
    else:
        atol = 0.06
        rtol = 0.06

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )
```

运行：

```bash
pytest -q tests/test_flash_decoding.py
```

再回归上一讲：

```bash
pytest -q tests/test_flash_attention.py
```

最后全量：

```bash
pytest -q
```

---

# 11. Step 11：手动验证一次 physical paging

假设：

```text
block_size = 4
context_len = 10
```

逻辑 sequence：

```text
token 0 1 2 3 | 4 5 6 7 | 8 9
```

而 block table：

```text
logical block 0 -> physical block 5
logical block 1 -> physical block 2
logical block 2 -> physical block 9
```

Stage 1：

```text
CTA(head, logical=0)
    ↓
table[0] = 5
    ↓
KCache[5, 0:4]

CTA(head, logical=1)
    ↓
table[1] = 2
    ↓
KCache[2, 0:4]

CTA(head, logical=2)
    ↓
table[2] = 9
    ↓
KCache[9, 0:2]
```

因此整个执行过程中都没有创建：

```text
k_hist[10, Hkv, D]
v_hist[10, Hkv, D]
```

这就是本讲相对 Advance 04 最大的数据路径变化。

---

# 12. Step 12：为什么 Stage 1 中每个 warp 还要先局部 online softmax

一个 CTA 有 4 个 warp。

例如：

```text
block_size = 16
```

每个 warp 分别看到：

```text
warp0: score0, score4, score8, score12
warp1: score1, score5, score9, score13
warp2: score2, score6, score10, score14
warp3: score3, score7, score11, score15
```

每个 warp 先独立得到：

```text
(m0, l0, O0)
(m1, l1, O1)
(m2, l2, O2)
(m3, l3, O3)
```

然后 CTA 内合并。

这比先 materialize：

```text
scores[16]
prob[16]
```

再做 softmax 更符合 FlashAttention / Flash-Decoding 的设计思想。

注意这里 `warp_reduce_sum_decoding()` 只用于：

```text
q · k
```

的 head-dim reduction。

softmax 本身没有使用 warp reduce，因为一个 warp 在本讲设计中是**顺序处理自己的 token 子序列**，直接进行 online update：

```text
score0
  ↓
(m,l,O)
  ↓
score4
  ↓
(m,l,O)
  ↓
score8
...
```

最后才跨 warp merge。

---

# 13. Step 13：为什么 Stage 2 仍然使用 4 个 warp

Stage 2 每个 q_head 有：

```text
num_kvcache_block
```

个 partial states。

本讲继续使用相同的 striping：

```text
warp0 -> block 0,4,8,...
warp1 -> block 1,5,9,...
warp2 -> block 2,6,10,...
warp3 -> block 3,7,11,...
```

每个 warp 先得到自己的：

```text
(m_w, l_w, O_w)
```

再 CTA 内合并 4 个 warp。

这样 Stage 2 不需要：

```text
thread0 串行遍历所有 blocks
```

也不需要第三个 kernel。

---

# 14. Step 14：为什么 `partial_O` 必须是 numerator，而不是局部最终输出

这是实现中最容易写错的地方之一。

Stage 1 对一个 partition 计算：

```text
O_block = Σ exp(score - m_block) * V
l_block = Σ exp(score - m_block)
```

此时不要保存：

```text
O_block / l_block
```

因为 Stage 2 合并时需要重新根据全局最大值调整每个 partition 的指数尺度。

正确 merge：

```text
m = max(m_a, m_b)

O = exp(m_a-m) * O_a
  + exp(m_b-m) * O_b

l = exp(m_a-m) * l_a
  + exp(m_b-m) * l_b
```

最后一次才：

```text
O /= l
```

因此本讲 Stage 1 的：

```text
partial_O
```

明确是 numerator accumulator。

---

# 15. Step 15：重新编译与完整验证流程

每次修改 `.cu` / `.cpp` 后重新安装：

```bash
MAX_JOBS=4 python -m pip install -e . --no-build-isolation
```

然后：

```bash
pytest -q tests/test_flash_attention.py
pytest -q tests/test_flash_decoding.py
pytest -q
```

再跑模型级测试：

```bash
TINYINFER_ATTENTION_BACKEND=torch \
python -m pytest -q tests/test_hf_equivalence.py
```

然后：

```bash
TINYINFER_ATTENTION_BACKEND=flash \
python -m pytest -q tests/test_hf_equivalence.py
```

如果两条路径文本或 logits 出现明显偏差，优先排查：

```text
1. context_len 是否包含当前 decode token。
2. current K/V 是否已经在 attention 前 store_kv。
3. block_table 是否传了当前 sequence 对应的那一行。
4. num_kvcache_block 是否使用 ceil(context_len / block_size)。
5. tail block 是否只访问 valid_tokens。
6. GQA q_head -> kv_head 映射是否正确。
7. partial_O 是否错误地提前除以 partial_l。
```

---

# 16. Step 16：如何与旧 decode 路径做性能对比

本讲要求保留原始 attention 实现，目的就是做下面这组对比。

旧路径：

```text
Paged KV
   ↓
gather_sequence_kv(K)
gather_sequence_kv(V)
   ↓
contiguous K/V
   ↓
FlashAttention
```

新路径：

```text
Paged KV
   ↓
Stage-1 Paged Flash-Decoding
   ↓
partial states
   ↓
Stage-2 reduction
```

测试时至少覆盖：

```text
context_len:
256
512
1024
2048
4096
8192
16384
```

以及：

```text
kvcache_block_size:
16
32
64
128
256
```

重点观察：

```text
1. decode attention kernel latency
2. TPOT
3. context_len 增长时的 scaling
4. 不同 KV block size 对 Stage-1 CTA 数的影响
5. Stage-2 reduction 在 block_size 很小时是否开始变重
```

当前版本：

```text
Flash-Decoding partition size
=
KV cache block size
```

因此：

```text
block_size 越小
→ Stage-1 CTA 越多
→ GPU parallelism 越高
→ partial states 越多
→ Stage-2 reduction 越重
```

这正是后续值得实测的 trade-off。

---

# 17. 当前版本的已知性能局限

本讲目标首先是把：

```text
Paged KV direct access
+
Flash-Decoding split-KV
```

完整接入 runtime，而不是一次做到最终性能。

当前至少还有以下优化空间。

## 17.1 `warp_O` 仍在 shared memory

当前：

```text
warp_O[4][head_dim]
```

放在 shared memory。

而对于常见：

```text
head_dim = 128
```

一个 lane 实际只维护：

```text
4 个 float accumulator
```

后续可以将 `HEAD_DIM` 模板化，让：

```text
O fragment
```

长期驻留 register。

## 17.2 KV page size 与 Flash-Decoding partition size 仍然绑定

当前：

```text
one CTA = one KV page
```

例如：

```text
block_size = 16
context_len = 16384
```

则：

```text
1024 partitions / head
```

Stage 2 需要合并大量 partial states。

后续更合理的是：

```text
KV_PAGE_SIZE = 16
TOKENS_PER_PARTITION = 128
```

即一个 CTA 跨多个 paged blocks。

## 17.3 Stage 2 仍然产生额外 GM traffic

Stage 1 需要写：

```text
partial_m
partial_l
partial_O
```

Stage 2 再读取。

这是 split-KV 用于换取 grid-level parallelism 的代价。

后续 benchmark 应寻找：

```text
single-CTA/head
vs
split-KV
```

的 context-length crossover point。

## 17.4 当前仍然逐 sequence 调 kernel

虽然 engine 支持 mixed batch，但 `Attention.forward()` 目前还是：

```python
for i in range(num_sequences):
    ...
```

因此多个 decode sequence 会分别 launch Flash-Decoding。

后续可以扩展成真正 batched paged decode：

```text
grid = [seq, head, partition]
```

并传入：

```text
context_lens[]
block_tables[][]
```

但这不属于本讲范围。

---

# 18. 本讲完成后的核心认知

Advance 04 的 FlashAttention 解决：

```text
连续 Q/K/V 上的 attention IO 问题
```

Advance 05 的 Flash-Decoding 进一步解决：

```text
q_len = 1 时：

1. 不再为了 attention 把 paged KV gather 成连续 Tensor。
2. 通过 block table 直接访问 physical KV blocks。
3. 把 KV sequence / KV blocks 暴露为 grid-level parallelism。
4. 使用两阶段 exact online-softmax merge 恢复最终 attention 输出。
```

整个演进关系变成：

```text
PagedKVCache
    │
    ├── prefill/chunked prefill
    │       ↓
    │   gather_sequence_kv
    │       ↓
    │   FlashAttention
    │
    └── decode q_len=1
            ↓
        block_table
            ↓
        Paged Flash-Decoding Stage 1
            ↓
        partial (m,l,O)
            ↓
        Stage 2 reduction
            ↓
        final O
```

最重要的一句话是：

> **FlashAttention 的 K/V tile 是 CTA 内循环；Flash-Decoding 把 K/V partition 提升成 grid 维度。多出来的第二阶段 reduction，正是用少量中间 GM traffic 换取更高 decode 并行度的代价。**

这也是本讲最核心的工业推理设计思想。
