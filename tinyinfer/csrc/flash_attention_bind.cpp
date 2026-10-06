#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// 定义于flash_attention.cu
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


// python端最终调用的函数
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



// 进行python函数注册; 注册函数名为flash_attention_forward, 参数列表为q, k, v, query_start_pos, scale, is_causal
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



