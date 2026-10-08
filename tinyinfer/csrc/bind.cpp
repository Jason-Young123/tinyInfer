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

// 定义于 flash_decoding.cu
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



// python 端 Paged Flash-Decoding 入口
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

    // 真实 logical-block 数由 context_len 决定，不能直接使用 padded table 长度
    const int64_t num_kvcache_block = (context_len + kvcache_block_size - 1) / kvcache_block_size;

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

    // Stage 1 中间状态全部使用 float32, 但是需要提前注册在GM上
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



