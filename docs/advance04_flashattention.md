# tinyInfer Advance 04：接入自定义 CUDA FlashAttention

这一讲直接基于当前 `tinyInfer-master(5)` 与 `Learning-CUDA-master` 继续修改。目标不是重写 attention runtime，而是在保留原始 PyTorch SDPA 路径的前提下，增加一条最小 CUDA FlashAttention 旁路，并让两条路径可以随时切换。

本讲只做到：

1. 保留 `tinyinfer/layers/attention.py` 中原始 PyTorch attention 计算逻辑。
2. 从 `Learning-CUDA/src/kernels.cu` 提取已有 FlashAttention 核心实现。
3. 将 `query_start_pos` 直接传入 CUDA，在 kernel 内实现 tinyInfer 所需的 causal mask。
4. PyTorch CUDA Tensor 直接把 device pointer 交给 CUDA kernel，不经过 CPU，不做 H2D/D2H 往返。
5. CUDA kernel 原生处理 GQA，不再为 FlashAttention 分支 `repeat_interleave(K/V)`。
6. 支持当前 tinyInfer 默认的 `torch.bfloat16`，同时保留 `float16/float32`。
7. 用环境变量在 `torch` 与 `flash` 两个 backend 之间切换。
8. 暂时不做 thread/block 参数调优、custom stream、batched-varlen kernel、直接读取 paged KV 等更深层优化。

本讲所有需要新增或修改的文件都给出**完整内容**。没有出现在本讲里的文件保持当前版本不变。

---

# 0. 最终目录变化

本讲完成后，新增/修改部分如下：

```text
tinyInfer/
├── setup.py                              # [NEW] 编译 PyTorch CUDA extension
├── tinyinfer/
│   ├── csrc/
│   │   ├── flash_attention.cu            # [NEW] CUDA kernel + launcher
│   │   └── flash_attention_bind.cpp      # [NEW] PyTorch C++ binding
│   └── layers/
│       ├── attention.py                  # [MOD] 增加最小 backend 旁路
│       └── flash_attention.py            # [NEW] Python 适配层
├── tests/
│   └── test_flash_attention.py           # [NEW] torch SDPA 对照测试
└── docs/
    └── advance04_flashattention.md
```

最终调用链：

```text
Qwen3Attention.forward
        ↓
Attention.forward
        ↓
store_kv + gather_sequence_kv
        ↓
Attention._attend_one
        │
        ├── backend=torch
        │      ↓
        │   原 PyTorch SDPA
        │
        └── backend=flash
               ↓
        flash_attention.py
               ↓
        flash_attention_bind.cpp
               ↓
        flash_attention.cu
               ↓
        kernel_flash_attention
```

注意：

```text
Scheduler / BlockManager / ModelRunner / Qwen3 / PagedKVCache
```

都不需要为了本次 FlashAttention 接入而修改。

---

# 1. Step 1：先固定两条 attention 路径的接口语义

当前 `Attention._attend_one()` 收到：

```text
q_i    : [Tq, Hq,  D]
k_hist : [Tk, Hkv, D]
v_hist : [Tk, Hkv, D]
```

其中：

```text
Tq  = 本轮这个 Sequence 真正计算的 token 数
Tk  = 本轮 attention 能看到的完整 KV 长度
Hq  = query heads
Hkv = key/value heads
D   = head_dim
```

另外还有：

```text
query_start_pos
```

它表示 `q_i[0]` 在完整 sequence 中的绝对位置。

因此：

```text
q_i[local_q]
```

对应的绝对位置是：

```text
query_start_pos + local_q
```

causal attention 的正确条件应为：

```text
k_pos <= query_start_pos + local_q
```

这条公式必须直接进入 CUDA kernel。

## 1.1 普通 prefill

例如：

```text
Tq = 64
Tk = 64
query_start_pos = 0
```

则：

```text
q[0] 只能看 k[0]
q[1] 可以看 k[0:2]
...
q[63] 可以看 k[0:64]
```

## 1.2 chunked prefill

例如已经有 48 个 token 的 KV，本轮再算 16 个：

```text
Tq = 16
Tk = 64
query_start_pos = 48
```

则：

```text
q[0]  的绝对位置 = 48，可看 k[0:49]
q[1]  的绝对位置 = 49，可看 k[0:50]
...
q[15] 的绝对位置 = 63，可看 k[0:64]
```

## 1.3 decode

例如：

```text
Tq = 1
Tk = 100
query_start_pos = 99
```

唯一的 query 应该看到：

```text
k[0:100]
```

因此不能继续使用原 Learning-CUDA 中：

```cpp
bid_z < j
bid_z == j && tid_y >= tid_x
```

这种“Q 和 K 都从 position 0 开始”的局部 block 判断。

这一版直接改成：

```cpp
int q_pos = query_start_pos + Br * bid_z + tid_y;
int k_pos = Bc * j + tid_x;
bool is_compute = !is_causal || (k_pos <= q_pos);
```

这样无需 Q-padding，也不会让 decode 从 `O(Tk)` 退化成无意义的 `O(Tk^2)` query 计算。

---

# 2. Step 2：新增 CUDA FlashAttention 实现

创建目录：

```bash
mkdir -p tinyinfer/csrc
```

创建：

```text
tinyinfer/csrc/flash_attention.cu
```

这一版基于 `Learning-CUDA/src/kernels.cu` 中现有 FlashAttention 代码做四个必要适配：

```text
1. 输入直接改为 device pointer，不再 cudaMalloc/cudaMemcpy。
2. 新增 query_start_pos，在 kernel 内完成绝对位置 causal mask。
3. 增加 BF16 输入/输出转换，匹配 tinyInfer 默认 torch.bfloat16。
4. 修复最后一个不完整 KV tile 时 O 更新线程不足的问题。
```

第 4 点非常重要。原代码 step-6 使用：

```cpp
if (tid_x < bound_tid_x && tid_y < bound_tid_y)
```

当最后一个 K/V tile 不满 32 个 token 时，`bound_tid_x < 32`，会导致部分 `tid_x` 不再参与输出 head_dim 的更新。这里应让所有 x 方向线程继续负责不同输出维度，因为无效的 P 元素已经被置零。

因此修改为：

```cpp
if (tid_y < bound_tid_y)
```

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


template <typename T>
__device__ inline float to_float(T x) {
    return static_cast<float>(x);
}


template <>
__device__ inline float to_float<half>(half x) {
    return __half2float(x);
}


template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}


template <typename T>
__device__ inline T from_float(float x) {
    return static_cast<T>(x);
}


template <>
__device__ inline half from_float<half>(float x) {
    return __float2half(x);
}


template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}


template <typename T>
__device__ inline T myexp(T x) {
    return exp(x);
}


template <>
__device__ inline float myexp<float>(float x) {
    return expf(x);
}


template <typename T>
__device__ inline T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


template <typename T>
__device__ inline T warp_reduce_max(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = val > tmp ? val : tmp;
    }
    return val;
}


template <typename T>
__global__ void kernel_flash_attention(
    int target_seq_len,
    int src_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int query_start_pos,
    float scale,
    bool is_causal,
    const T* Q,
    const T* K,
    const T* V,
    T* O
) {
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int q_head = blockIdx.x;
    const int q_block = blockIdx.z;

    const int Br = blockDim.y;
    const int Bc = blockDim.x;
    const int Tc = (src_seq_len + Bc - 1) / Bc;
    const int q_per_kv = q_heads / kv_heads;
    const int kv_head = q_head / q_per_kv;

    extern __shared__ char shared_mem[];
    char* ptr = shared_mem;

    double* SP = reinterpret_cast<double*>(ptr);
    ptr += Br * Bc * sizeof(double);

    float* m_prev = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* m_new = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* l_prev = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* l_new = reinterpret_cast<float*>(ptr);
    ptr += Br * sizeof(float);

    float* Q_sm = reinterpret_cast<float*>(ptr);
    ptr += Br * head_dim * sizeof(float);

    float* K_T_sm = reinterpret_cast<float*>(ptr);
    ptr += head_dim * Bc * sizeof(float);

    float* V_sm = reinterpret_cast<float*>(ptr);
    ptr += Bc * head_dim * sizeof(float);

    float* O_sm = reinterpret_cast<float*>(ptr);

#define SP_AT(y, x) SP[(y) * Bc + (x)]
#define Q_AT(y, x) Q_sm[(y) * head_dim + (x)]
#define KT_AT(y, x) K_T_sm[(y) * Bc + (x)]
#define V_AT(y, x) V_sm[(y) * head_dim + (x)]
#define O_AT(y, x) O_sm[(y) * head_dim + (x)]

    const int q_block_start = Br * q_block;
    const int bound_q = min(Br, target_seq_len - q_block_start);
    const int q_block_local_end = min(target_seq_len - 1, q_block_start + Br - 1);
    const int q_block_abs_end = query_start_pos + q_block_local_end;

    for (int d = tid_x; d < head_dim; d += blockDim.x) {
        O_AT(tid_y, d) = 0.0f;
        Q_AT(tid_y, d) = 0.0f;
        if (tid_y < bound_q) {
            const int q_idx = q_block_start + tid_y;
            const int offset = (q_idx * q_heads + q_head) * head_dim + d;
            Q_AT(tid_y, d) = to_float<T>(Q[offset]);
        }
    }
    __syncthreads();

    if (tid_x == 0) {
        m_prev[tid_y] = -8192.0f;
        l_prev[tid_y] = 0.0f;
    }
    __syncthreads();

#pragma unroll 4
    for (int j = 0; j < Tc; ++j) {
        const int k_block_start = Bc * j;

        if (is_causal && k_block_start > q_block_abs_end) {
            break;
        }

        SP_AT(tid_y, tid_x) = -8192.0;
        __syncthreads();

        const int bound_k = min(Bc, src_seq_len - k_block_start);
        const int q_pos = query_start_pos + q_block_start + tid_y;
        const int k_pos = k_block_start + tid_x;
        const bool is_compute = !is_causal || (k_pos <= q_pos);

#pragma unroll
        for (int d = tid_x; d < head_dim; d += blockDim.x) {
            KT_AT(d, tid_y) = 0.0f;
            V_AT(tid_y, d) = 0.0f;

            if (tid_y < bound_k) {
                const int k_idx = k_block_start + tid_y;
                const int offset = (k_idx * kv_heads + kv_head) * head_dim + d;
                KT_AT(d, tid_y) = to_float<T>(K[offset]);
                V_AT(tid_y, d) = to_float<T>(V[offset]);
            }
        }
        __syncthreads();

        if (tid_y < bound_q && tid_x < bound_k && is_compute) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                dot += Q_AT(tid_y, d) * KT_AT(d, tid_x);
            }
            SP_AT(tid_y, tid_x) = static_cast<double>(dot) * static_cast<double>(scale);
        }
        __syncthreads();

        float row_max = static_cast<float>(SP_AT(tid_y, tid_x));
        row_max = warp_reduce_max(row_max);

        if (tid_x == 0 && tid_y < bound_q) {
            m_new[tid_y] = row_max > m_prev[tid_y] ? row_max : m_prev[tid_y];
        }
        __syncthreads();

        if (tid_y < bound_q && tid_x < bound_k && is_compute) {
            SP_AT(tid_y, tid_x) =
                myexp<double>(SP_AT(tid_y, tid_x) - static_cast<double>(m_new[tid_y]));
        } else {
            SP_AT(tid_y, tid_x) = 0.0;
        }
        __syncthreads();

        float row_sum = static_cast<float>(SP_AT(tid_y, tid_x));
        row_sum = warp_reduce_sum(row_sum);

        float prev_scale = 0.0f;
        if (tid_y < bound_q) {
            prev_scale = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
        }

        if (tid_x == 0 && tid_y < bound_q) {
            l_new[tid_y] = prev_scale * l_prev[tid_y] + row_sum;
        }
        __syncthreads();

        if (tid_y < bound_q) {
            for (int d = tid_x; d < head_dim; d += blockDim.x) {
                float pv = 0.0f;
#pragma unroll
                for (int w = 0; w < Bc; ++w) {
                    pv += static_cast<float>(SP_AT(tid_y, w)) * V_AT(w, d);
                }
                O_AT(tid_y, d) = O_AT(tid_y, d) * prev_scale + pv;
            }
        }
        __syncthreads();

        if (tid_x == 0 && tid_y < bound_q) {
            m_prev[tid_y] = m_new[tid_y];
            l_prev[tid_y] = l_new[tid_y];
        }
        __syncthreads();
    }

#pragma unroll
    for (int d = tid_x; d < head_dim; d += blockDim.x) {
        if (tid_y < bound_q) {
            const int q_idx = q_block_start + tid_y;
            const int offset = (q_idx * q_heads + q_head) * head_dim + d;
            O[offset] = from_float<T>(O_AT(tid_y, d) / l_prev[tid_y]);
        }
    }

#undef SP_AT
#undef Q_AT
#undef KT_AT
#undef V_AT
#undef O_AT
}


template <typename T>
void launch_flash_attention_typed(
    const T* q,
    const T* k,
    const T* v,
    T* out,
    int target_seq_len,
    int src_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int query_start_pos,
    float scale,
    bool is_causal
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;

    const int grid_z = (target_seq_len + Br - 1) / Br;
    const dim3 block_dim(Bc, Br);
    const dim3 grid_dim(q_heads, 1, grid_z);

    const size_t smem_size =
        Br * Bc * sizeof(double) +
        Br * 4 * sizeof(float) +
        (Br * head_dim * 2 + Bc * head_dim * 2) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel_flash_attention<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)
    ));

    kernel_flash_attention<T><<<grid_dim, block_dim, smem_size>>>(
        target_seq_len,
        src_seq_len,
        q_heads,
        kv_heads,
        head_dim,
        query_start_pos,
        scale,
        is_causal,
        q,
        k,
        v,
        out
    );

    CUDA_CHECK(cudaGetLastError());
}


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
) {
    if (dtype_code == 0) {
        launch_flash_attention_typed<half>(
            static_cast<const half*>(q),
            static_cast<const half*>(k),
            static_cast<const half*>(v),
            static_cast<half*>(out),
            target_seq_len,
            src_seq_len,
            q_heads,
            kv_heads,
            head_dim,
            query_start_pos,
            scale,
            is_causal
        );
        return;
    }

    if (dtype_code == 1) {
        launch_flash_attention_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),
            static_cast<__nv_bfloat16*>(out),
            target_seq_len,
            src_seq_len,
            q_heads,
            kv_heads,
            head_dim,
            query_start_pos,
            scale,
            is_causal
        );
        return;
    }

    if (dtype_code == 2) {
        launch_flash_attention_typed<float>(
            static_cast<const float*>(q),
            static_cast<const float*>(k),
            static_cast<const float*>(v),
            static_cast<float*>(out),
            target_seq_len,
            src_seq_len,
            q_heads,
            kv_heads,
            head_dim,
            query_start_pos,
            scale,
            is_causal
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code");
}
```

## 2.1 为什么这里不再保留原 host `flashAttention(std::vector<T>...)`

Learning-CUDA 原接口是：

```text
std::vector on CPU
    ↓ cudaMemcpy H2D
CUDA kernel
    ↓ cudaMemcpy D2H
std::vector on CPU
```

而 tinyInfer 的 Q/K/V 本来就已经在 GPU：

```text
PyTorch CUDA Tensor
```

如果继续复用原 host wrapper，实际会变成：

```text
GPU → CPU → GPU → CPU → GPU
```

这会完全破坏 attention 的运行时性能。

因此本次保留的是：

```text
FlashAttention kernel 的核心算法
```

而不是保留测试程序使用的 host-memory wrapper。

## 2.2 为什么仍然保留固定 `Br=Bc=32`

这一讲只做接口接通与 correctness。

当前继续使用 Learning-CUDA 原来的：

```cpp
Br = 32;
Bc = 32;
```

即一个 block：

```text
32 × 32 = 1024 threads
```

线程块选择与更深层 kernel tuning 留到后续，不在这里展开。

## 2.3 为什么要调用 `cudaFuncSetAttribute`

Qwen3 常见 `head_dim=128` 时，本 kernel 动态 shared memory 大约为：

```text
SP                    32 × 32 × 8
m/l                    32 × 4 × 4
Q/K/V/O                4 × 32 × 128 × 4
--------------------------------------
约 74 KB
```

这可能超过 CUDA 默认允许的 dynamic shared-memory 阈值，因此需要显式 opt-in。

这不是性能调优，而是为了让当前 kernel 在较大 `head_dim` 下能够正常 launch。

---

# 3. Step 3：增加 PyTorch C++ Binding

创建：

```text
tinyinfer/csrc/flash_attention_bind.cpp
```

这一层只负责：

```text
PyTorch Tensor
    ↓ 检查 shape/device/dtype
裸 device pointer
    ↓
flash_attention_forward_cuda(...)
```

不在这里实现 attention 数学。

完整文件：

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


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
}
```

这里没有：

```text
Tensor.cpu()
std::vector
cudaMalloc Q/K/V
cudaMemcpy H2D
cudaMemcpy D2H
```

因此 Q/K/V 数据路径是：

```text
PyTorch CUDA Tensor
        ↓ data_ptr()
CUDA kernel
        ↓
PyTorch CUDA Tensor
```

---

# 4. Step 4：增加 Python FlashAttention 适配层

创建：

```text
tinyinfer/layers/flash_attention.py
```

这一层的职责很薄：

```text
1. 延迟导入编译后的 tinyinfer._C。
2. 保证 Q/K/V contiguous。
3. 把 query_start_pos 与 scale 传给 C++ extension。
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


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_start_pos: int,
    scale: float,
) -> torch.Tensor:
    if q.device.type != "cuda":
        raise ValueError("FlashAttention requires CUDA tensors")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    return _load_extension().flash_attention_forward(
        q,
        k,
        v,
        int(query_start_pos),
        float(scale),
        True,
    )
```

注意 `.contiguous()` 不代表每次都会复制。

如果 Tensor 本身已经 contiguous：

```python
q.contiguous()
```

直接返回原 Tensor view，不会重新分配数据。

当前 `q_i` 是首维 slice，`k_hist/v_hist` 来自 `torch.cat`，正常情况下本来就是 contiguous，因此这里主要是接口保险。

---

# 5. Step 5：在 Attention 中加入最小旁路

修改：

```text
tinyinfer/layers/attention.py
```

设计原则：

```text
原 torch attention 实现不重写；
只在 _attend_one() 最前面加一个 early branch。
```

backend 使用环境变量：

```text
TINYINFER_ATTENTION_BACKEND=torch
TINYINFER_ATTENTION_BACKEND=flash
```

这样无需修改 `Config`、`Qwen3Attention` 或 `ModelRunner`。

下面是修改后的**完整文件**：

```python
import os

import torch
from torch import nn

from tinyinfer.utils.context import get_context


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
    # k/v: [num_tokens, num_kv_heads, head_dim]
    if k.shape != v.shape:
        raise ValueError("k/v shape mismatch")
    if k.shape[0] != slot_mapping.numel():
        raise ValueError("slot_mapping length must equal token count")
    for i, slot in enumerate(slot_mapping.tolist()):
        block = slot // block_size
        offset = slot % block_size
        cache_k[block, offset].copy_(k[i])
        cache_v[block, offset].copy_(v[i])


# 针对某个请求 seq 获取其全部 context 对应的 K/V cache
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

        # [MOD] runtime backend，不污染模型 config。
        self.backend = os.getenv("TINYINFER_ATTENTION_BACKEND", "torch").strip().lower()
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

    # unified attention computation
    def _attend_one(self, q_i, k_hist, v_hist, query_start_pos: int):
        # q_i:    [Tq, Hq, D]
        # k_hist: [Tk, Hkv, D]

        # [MOD] FlashAttention 直接处理原始 GQA heads 与 query_start_pos。
        if self.backend == "flash":
            from tinyinfer.layers.flash_attention import flash_attention

            return flash_attention(
                q_i,
                k_hist,
                v_hist,
                query_start_pos=query_start_pos,
                scale=self.scale,
            )

        # 原始 torch SDPA 路径保持不变。
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

    # interface
    def forward(self, q, k, v):
        ctx = get_context()
        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx)

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

## 5.1 为什么 Flash 分支不调用 `_repeat()`

当前 torch SDPA 路径需要：

```text
K/V heads → repeat 到 Q heads
```

例如：

```text
Hq  = 16
Hkv = 8
```

会把 K/V 从 8 heads materialize 成 16 heads。

而 CUDA kernel 已经直接计算：

```cpp
const int q_per_kv = q_heads / kv_heads;
const int kv_head = q_head / q_per_kv;
```

即：

```text
Q head 0,1 → KV head 0
Q head 2,3 → KV head 1
...
```

因此 FlashAttention 分支直接接受：

```text
Q : [Tq, Hq, D]
K : [Tk, Hkv, D]
V : [Tk, Hkv, D]
```

不做额外 K/V repeat。

---

# 6. Step 6：增加 CUDA Extension 构建入口

在仓库根目录新增：

```text
setup.py
```

完整内容：

```python
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
CSRC = ROOT / "tinyinfer" / "csrc"


setup(
    ext_modules=[
        CUDAExtension(
            name="tinyinfer._C",
            sources=[
                str(CSRC / "flash_attention_bind.cpp"),
                str(CSRC / "flash_attention.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
```

`pyproject.toml` 本讲不需要修改。

由于当前 `pyproject.toml` 的 isolated build environment 不保证预装 PyTorch，因此构建 extension 时使用：

```bash
MAX_JOBS=4 python -m pip install -e . --no-build-isolation
```

这里的：

```text
--no-build-isolation
```

让 build 过程直接使用当前环境中已经安装好的：

```text
torch
CUDA toolkit
```

---

# 7. Step 7：先检查 CUDA 编译环境

在 tinyInfer 根目录：

```bash
nvcc --version
```

然后：

```bash
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("CUDA_HOME:", CUDA_HOME)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

需要至少确认：

```text
cuda available: True
CUDA_HOME: 不是 None
```

然后编译：

```bash
MAX_JOBS=4 python -m pip install -e . --no-build-isolation
```

验证 extension：

```bash
python - <<'PY'
from tinyinfer import _C
print(_C)
print(hasattr(_C, "flash_attention_forward"))
PY
```

预期最后一行：

```text
True
```

---

# 8. Step 8：增加独立 correctness test

不要第一步就跑完整 Qwen3 文本生成。

先单独验证：

```text
自定义 FlashAttention
vs
PyTorch scaled_dot_product_attention
```

创建：

```text
tests/test_flash_attention.py
```

完整文件：

```python
import pytest
import torch
import torch.nn.functional as F

from tinyinfer.layers.flash_attention import flash_attention


CUDA_AVAILABLE = torch.cuda.is_available()


def torch_reference(q, k, v, query_start_pos, scale):
    hq = q.shape[1]
    hkv = k.shape[1]

    q_ref = q.transpose(0, 1).unsqueeze(0)
    k_ref = k.transpose(0, 1).unsqueeze(0)
    v_ref = v.transpose(0, 1).unsqueeze(0)

    if hq != hkv:
        repeat = hq // hkv
        k_ref = k_ref.repeat_interleave(repeat, dim=1)
        v_ref = v_ref.repeat_interleave(repeat, dim=1)

    tq = q.shape[0]
    tk = k.shape[0]

    q_pos = torch.arange(
        query_start_pos,
        query_start_pos + tq,
        device=q.device,
    )
    k_pos = torch.arange(tk, device=q.device)

    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    causal = causal.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=causal,
        is_causal=False,
        scale=scale,
    )

    return out.squeeze(0).transpose(0, 1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tq,tk,query_start_pos",
    [
        (37, 37, 0),
        (7, 37, 30),
        (1, 37, 36),
    ],
)
def test_flash_attention_matches_torch(dtype, tq, tk, query_start_pos):
    torch.manual_seed(0)

    hq = 8
    hkv = 2
    head_dim = 128
    scale = head_dim ** -0.5

    q = torch.randn(
        tq,
        hq,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1
    k = torch.randn(
        tk,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1
    v = torch.randn(
        tk,
        hkv,
        head_dim,
        device="cuda",
        dtype=dtype,
    ) * 0.1

    expected = torch_reference(
        q,
        k,
        v,
        query_start_pos=query_start_pos,
        scale=scale,
    )
    actual = flash_attention(
        q,
        k,
        v,
        query_start_pos=query_start_pos,
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

这里故意选择：

```text
Tk = 37
```

而不是 32/64。

这样会直接覆盖：

```text
最后一个 K/V tile 不满 32
```

的边界情况，可以验证前面 step-6 输出更新条件的修复。

三个 case 分别代表：

```text
(37, 37, 0)   → 普通 prefill
(7, 37, 30)   → chunked prefill
(1, 37, 36)   → decode
```

同时：

```text
Hq = 8
Hkv = 2
```

也会验证 GQA。

运行：

```bash
pytest -q tests/test_flash_attention.py -s
```

预期：

```text
6 passed
```

如果出现小量数值误差，不要期待 bitwise equal。

原因是两个实现的：

```text
reduction 顺序
online softmax 顺序
中间精度
```

并不完全相同。

当前测试目标是：

```text
数值在合理 tolerance 内一致
```

而不是二进制完全一致。

---

# 9. Step 9：验证原 torch backend 完全没有被破坏

默认什么环境变量都不设置：

```bash
unset TINYINFER_ATTENTION_BACKEND
```

此时：

```python
self.backend == "torch"
```

继续走原始：

```text
_repeat(K/V)
    ↓
构造 causal mask
    ↓
torch.nn.functional.scaled_dot_product_attention
```

运行原有测试：

```bash
pytest -q
```

这一阶段首先确认：

```text
原 tinyInfer 行为没有因为 FlashAttention 接入而改变
```

---

# 10. Step 10：切换到 FlashAttention backend

设置：

```bash
export TINYINFER_ATTENTION_BACKEND=flash
```

验证：

```bash
python - <<'PY'
import os
print(os.getenv("TINYINFER_ATTENTION_BACKEND"))
PY
```

应输出：

```text
flash
```

此时每个 `Attention` 实例会在构造时记录：

```python
self.backend = "flash"
```

后续 `_attend_one()` 走：

```text
q_i/k_hist/v_hist
    ↓
flash_attention.py
    ↓
tinyinfer._C
    ↓
CUDA FlashAttention kernel
```

切回原始实现：

```bash
export TINYINFER_ATTENTION_BACKEND=torch
```

或者：

```bash
unset TINYINFER_ATTENTION_BACKEND
```

---

# 11. Step 11：为什么 mixed batching 不需要额外修改

当前 `Attention.forward()` 已经根据：

```python
cu_q = ctx.cu_seqlens_q.tolist()
```

逐个 Sequence 拆分：

```python
for i in range(len(cu_q) - 1):
    ...
    q_i = q[qs:qe]
```

因此一个 mixed batch 即使同时包含：

```text
seq0: chunked prefill, Tq=16
seq1: decode,          Tq=1
seq2: prefill,         Tq=8
```

也会变成三次独立调用：

```text
_attend_one(seq0)
_attend_one(seq1)
_attend_one(seq2)
```

FlashAttention 当前只需要处理单个 Sequence：

```text
Q : [Tq, Hq, D]
K : [Tk, Hkv, D]
V : [Tk, Hkv, D]
```

因此暂时不需要自己实现：

```text
cu_seqlens
varlen batch
```

这会牺牲一部分 kernel launch 效率，但可以最大限度保持当前 runtime 不变。

后续如果再做性能版本，可以把多个 Sequence 一次送给 varlen FlashAttention kernel；那是下一阶段，而不是本讲的目标。

---

# 12. Step 12：为什么 Paged KV Cache 也不需要修改

当前 attention 路径已经先做：

```python
k_hist = gather_sequence_kv(...)
v_hist = gather_sequence_kv(...)
```

因此 kernel 看到的是连续：

```text
K : [Tk, Hkv, D]
V : [Tk, Hkv, D]
```

而不是物理 KV block。

当前数据流：

```text
PagedKVCache
    ↓ gather_sequence_kv
contiguous K/V
    ↓
FlashAttention
```

这意味着：

```text
FlashAttention
```

和：

```text
Paged KV allocator
```

仍然保持解耦。

本讲不做：

```text
FlashAttention kernel 直接按照 block_table 读取 paged KV
```

因为那已经属于 PagedAttention / paged FlashAttention 范畴，会显著扩大改动面。

---

# 13. Step 13：完整理解 CUDA causal mask

这是本讲最重要的语义变化。

kernel 中：

```cpp
const int q_pos = query_start_pos + q_block_start + tid_y;
const int k_pos = k_block_start + tid_x;
const bool is_compute = !is_causal || (k_pos <= q_pos);
```

## 13.1 chunked prefill 示例

假设：

```text
query_start_pos = 48
Tq = 16
Tk = 64
```

当前 Q block：

```text
q_block_start = 0
```

对于：

```text
tid_y = 0
```

有：

```text
q_pos = 48
```

所以：

```text
k_pos <= 48
```

才能参与。

对于：

```text
tid_y = 15
```

有：

```text
q_pos = 63
```

因此：

```text
k_pos <= 63
```

全部 64 个 KV 都可见。

## 13.2 decode 示例

```text
query_start_pos = 99
Tq = 1
Tk = 100
```

唯一的 query：

```text
q_pos = 99
```

所有：

```text
k_pos = 0...99
```

都满足：

```text
k_pos <= q_pos
```

因此 decode 可以正确读取全部历史 KV。

## 13.3 为什么不再需要 Q-padding

如果不修改 kernel，可以把 Q 补成：

```text
[0, 0, ..., real_q]
```

让 real Q 落在真实绝对位置。

但 decode 时：

```text
Tq = 1
Tk = 4096
```

会被迫构造并计算 4096 个 query rows。

现在直接传：

```text
query_start_pos = 4095
Tq = 1
```

kernel 只计算真正的 1 个 query row。

因此当前方案同时兼顾：

```text
移植简单
+
避免明显的 decode 冗余计算
```

---

# 14. Step 14：这一版的数据搬运到底还有哪些

从 Qwen3Attention 得到：

```text
q/k/v
```

它们已经是 CUDA Tensor。

本轮新 K/V 先写入：

```text
PagedKVCache
```

随后当前代码仍然通过：

```python
gather_sequence_kv(...)
```

把物理 block gather 成连续 `k_hist/v_hist`。

因此目前还存在：

```text
Paged KV → contiguous K/V
```

这个 GPU 内部的数据整理过程。

但已经消除了最不合理的：

```text
GPU → CPU → GPU
```

往返。

所以当前版本的数据路径是：

```text
Paged KV on GPU
    ↓ gather on GPU
contiguous K/V on GPU
    ↓ device pointer
FlashAttention kernel
    ↓
output Tensor on GPU
```

这是当前“尽量少改 tinyInfer”条件下合理的边界。

---

# 15. Step 15：为什么默认 BF16 必须支持

当前 `ModelRunner` 构造函数默认：

```python
def __init__(
    self,
    config,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
```

因此整个模型与 KV cache 默认都会是：

```text
torch.bfloat16
```

如果 CUDA extension 只支持 Learning-CUDA 原来的：

```text
float
half
```

则完整 tinyInfer 一切换到 flash backend 就会失败。

所以本讲增加：

```cpp
__nv_bfloat16
```

输入输出支持。

kernel 内部仍然把 Q/K/V tile 转为：

```text
float
```

进行主体计算，因此只是增加边界 dtype 适配，并没有重写 FlashAttention 数学。

---

# 16. Step 16：当前版本明确没有做什么

为了控制改动范围，这一讲刻意不做以下优化。

## 16.1 不调 block/thread 布局

继续：

```text
Br = 32
Bc = 32
block = 32 × 32
```

## 16.2 不接 PyTorch current stream

当前 kernel 使用 CUDA default stream。

当前 tinyInfer 也没有建立 custom stream runtime，因此这一版先保持简单。

后续如果引入：

```text
多 stream
异步 H2D
overlap
```

再让 extension 显式获取 PyTorch current CUDA stream。

## 16.3 不做多 Sequence varlen kernel

现在仍然：

```text
一个 Sequence → 一次 FlashAttention launch
```

## 16.4 不让 kernel 直接读 paged KV

当前仍然：

```text
gather_sequence_kv
    ↓
contiguous K/V
```

## 16.5 不做 backward

这是 tinyInfer inference runtime，因此只实现：

```text
forward
```

不注册 autograd backward。

---

# 17. Step 17：建议的验证顺序

不要跳步骤。

第一层：

```bash
MAX_JOBS=4 python -m pip install -e . --no-build-isolation
```

确认 extension 能编译和 import。

第二层：

```bash
pytest -q tests/test_flash_attention.py -s
```

确认：

```text
prefill
chunked prefill
decode
GQA
BF16/FP16
非 32 对齐长度
```

全部通过。

第三层：

```bash
unset TINYINFER_ATTENTION_BACKEND
pytest -q
```

确认原 torch backend 没被破坏。

第四层：

```bash
export TINYINFER_ATTENTION_BACKEND=flash
```

再跑你当前真实 Qwen3 推理脚本。

第五层再比较：

```text
torch backend 输出 token
vs
flash backend 输出 token
```

如果偶尔出现 sampling 差异，先不要直接判断 kernel 错误。

应该优先比较：

```text
单层 attention output
hidden states
logits
```

的数值误差，因为 sampling 可能把很小的 logits 差异放大成不同 token。

---

# 18. Step 18：本讲完成后的 architecture checkpoint

完成 Advance 04 后，你的 tinyInfer attention 路径已经从：

```text
Qwen3
  ↓
Paged KV Cache
  ↓
PyTorch SDPA
```

变成：

```text
                       ┌── PyTorch SDPA
Qwen3 → Paged KV → ────┤
                       └── 自己的 CUDA FlashAttention
```

而且两条路径共享完全相同的：

```text
Scheduler
Sequence metadata
Prefix Cache
Paged KV Cache
Mixed Batching
Qwen3 model
```

差异只发生在：

```text
Attention._attend_one()
```

这一层。

这正是当前阶段最重要的工程边界：

> runtime 负责“本轮哪些 token 要算、历史 KV 在哪里”；attention backend 只负责“给定 Q 与完整 K/V，如何算出 attention output”。

当前 FlashAttention backend 已经做到：

```text
GPU Tensor 直连 CUDA
GQA 原生处理
chunked prefill 正确 causal mask
decode 正确 causal mask
FP16/BF16/FP32 接口
原 torch 路径完整保留
```

下一阶段如果继续优化，才值得依次研究：

```text
PyTorch current stream
↓
更合理的 thread/block 配置
↓
减少 gather_sequence_kv
↓
varlen mixed-batch FlashAttention
↓
paged KV direct access
```

但这些都建立在本讲 correctness-first 的旁路已经稳定工作之后。
