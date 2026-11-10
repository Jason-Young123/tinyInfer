#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


/**************************** defined in: flash_attention.cu ****************************/
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







/**************************** defined in: flash_decoding.cu ****************************/
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









/**************************** defined in: rmsnorm.cu ****************************/
extern "C" void rmsnorm_forward_cuda(
    const void* input,
    size_t hidden_dim,
    size_t num_rows,
    const void* weight,
    float eps,
    void* output,
    int dtype_code
);


torch::Tensor rmsnorm_forward(
    torch::Tensor input, //shape = [num_tokens, num_heads, hidden_dim] or [num_tokens, hidden_dim']
    torch::Tensor weight,//shape = [hidden_dim]
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.device() == weight.device(), "input/weight must be on the same device");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must have shape [D]");
    TORCH_CHECK(input.size(-1) == weight.size(0), "weight size must match input last dimension");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input/weight dtype mismatch");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int dtype_code = -1;
    if (input.scalar_type() == at::kHalf) {
        dtype_code = 0;
    } else if (input.scalar_type() == at::kBFloat16) {
        dtype_code = 1;
    } else if (input.scalar_type() == at::kFloat) {
        dtype_code = 2;
    } else {
        TORCH_CHECK(false, "RMSNorm supports fp16, bf16 and fp32 only");
    }

    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    size_t hidden_dim = input.size(-1);
    size_t num_rows = input.numel() / hidden_dim;
    rmsnorm_forward_cuda(
        input.data_ptr(),
        hidden_dim,
        num_rows,
        weight.data_ptr(),
        eps,
        output.data_ptr(),
        dtype_code
    );
    
    return output;
}






/**************************** defined in: rope.cu ****************************/
extern "C" void rope_forward_cuda(
    const void* q,         // [T, Hq, D]
    const void* k,         // [T, Hk, D]
    const int64_t* positions, // [T]
    const float* inv_freq, // [RD/2]
    int hidden_dim,     // D
    int rotary_dim,     // RD
    int num_tokens,     // T
    int head_q,         // Hq
    int head_k,         // Hk
    void* q_rot,           // [T, Hq, D]
    void* k_rot,            // [T, Hk, D]
    int dtype_code
);


std::tuple<torch::Tensor, torch::Tensor> rope_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor positions,
    torch::Tensor inv_freq
) {
    // ---------------- Device checks ----------------
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");

    TORCH_CHECK(
        q.device() == k.device() &&
        q.device() == positions.device() &&
        q.device() == inv_freq.device(),
        "all tensors must be on the same device"
    );

    // ---------------- Shape checks ----------------
    TORCH_CHECK(q.dim() == 3, "q must have shape [T, Hq, D]");
    TORCH_CHECK(k.dim() == 3, "k must have shape [T, Hk, D]");
    TORCH_CHECK(positions.dim() == 1, "positions must have shape [T]");
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must have shape [RD/2]");

    TORCH_CHECK(q.size(0) == k.size(0), "q/k token count mismatch");
    TORCH_CHECK(q.size(0) == positions.numel(), "positions length mismatch");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k head_dim mismatch");

    TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0, "head count must be positive");
    TORCH_CHECK(q.size(2) > 0, "hidden_dim must be positive");
    TORCH_CHECK(inv_freq.numel() > 0, "inv_freq must not be empty");

    // ---------------- Dtype checks ----------------
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "inv_freq must be float32");

    int dtype_code = -1;

    if (q.scalar_type() == at::kHalf) {
        dtype_code = 0;
    } else if (q.scalar_type() == at::kBFloat16) {
        dtype_code = 1;
    } else if (q.scalar_type() == at::kFloat) {
        dtype_code = 2;
    } else {
        TORCH_CHECK(false, "RoPE supports fp16, bf16 and fp32 only");
    }

    // ---------------- Contiguous checks ----------------
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(inv_freq.is_contiguous(), "inv_freq must be contiguous");

    // ---------------- Kernel parameters ----------------
    int num_tokens = static_cast<int>(q.size(0));
    int head_q = static_cast<int>(q.size(1));
    int head_k = static_cast<int>(k.size(1));
    int hidden_dim = static_cast<int>(q.size(2));

    // RD = 2 * inv_freq.numel()
    int rotary_dim = static_cast<int>(inv_freq.numel() * 2);

    TORCH_CHECK(rotary_dim <= hidden_dim, "rotary_dim cannot exceed hidden_dim");

    // ---------------- Device guard ----------------
    c10::cuda::CUDAGuard device_guard(q.device());

    // ---------------- Allocate outputs ----------------
    auto q_rot = torch::empty_like(q);
    auto k_rot = torch::empty_like(k);

    // Empty input: no kernel launch
    if (num_tokens == 0) {
        return std::make_tuple(q_rot, k_rot);
    }

    // ---------------- Launch CUDA kernel ----------------
    rope_forward_cuda(
        q.data_ptr(),
        k.data_ptr(),
        positions.data_ptr<int64_t>(),
        inv_freq.data_ptr<float>(),
        hidden_dim,
        rotary_dim,
        num_tokens,
        head_q,
        head_k,
        q_rot.data_ptr(),
        k_rot.data_ptr(),
        dtype_code
    );

    return std::make_tuple(q_rot, k_rot);
}






/**************************** defined in: linear.cu ****************************/
extern "C" void linear_forward_cutlass_(
    const void* X,
    const void* W,
    size_t M,
    size_t K,
    size_t N,
    void* O,
    int dtype_code
);

torch::Tensor linear_forward_cutlass(
    torch::Tensor X,
    torch::Tensor W
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "X and W must be CUDA tensors");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "X and W must be 2D tensors");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous(), "X and W must be contiguous");
    TORCH_CHECK(X.device() == W.device(), "X and W must be on the same device");
    TORCH_CHECK(X.scalar_type() == W.scalar_type(), "X and W must have the same dtype");
    TORCH_CHECK(X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16, "Only FP16 and BF16 are supported");

    auto M = X.size(0);
    auto K = X.size(1);
    auto N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "K dimension mismatch");

    c10::cuda::CUDAGuard device_guard(X.device());
    auto O = torch::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)}, X.options());

    int dtype_code;
    if (X.scalar_type() == at::kHalf) {
        dtype_code = 0;
    }
    else {
        dtype_code = 1;
    }

    linear_forward_cutlass_(
        X.data_ptr(),
        W.data_ptr(),
        M,
        K,
        N,
        O.data_ptr(),
        dtype_code
    );

    return O;
}



/**************************** defined in: linear.cu ****************************/
extern "C" int linear_forward_cublas_(
    const void* X,
    const void* W,
    size_t M,
    size_t K,
    size_t N,
    void* O,
    int dtype_code
); 

torch::Tensor linear_forward_cublas(
    torch::Tensor X,
    torch::Tensor W
) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "X and W must be CUDA tensors");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "X and W must be 2D tensors");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous(), "X and W must be contiguous");
    TORCH_CHECK(X.device() == W.device(), "X and W must be on the same device");
    TORCH_CHECK(X.scalar_type() == W.scalar_type(), "X and W must have the same dtype");
    TORCH_CHECK(X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16, "Only FP16 and BF16 are supported");

    auto M = X.size(0);
    auto K = X.size(1);
    auto N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "K dimension mismatch");

    c10::cuda::CUDAGuard device_guard(X.device());
    auto O = torch::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)}, X.options());

    int dtype_code;
    if (X.scalar_type() == at::kHalf) {
        dtype_code = 0;
    }
    else {
        dtype_code = 1;
    }

    linear_forward_cublas_(
        X.data_ptr(),
        W.data_ptr(),
        M,
        K,
        N,
        O.data_ptr(),
        dtype_code
    );

    return O;
}




/**************************** defined in: mlp.cu ****************************/
extern "C" void mlp_forward_cuda(
    const void* X,     // [N, hidden_dim]
    const void* gate,  // [intermediate, hidden_dim]
    const void* up,    // [intermediate, hidden_dim]
    const void* down,  // [hidden_dim, intermediate]
    size_t N,
    size_t hidden_dim,
    size_t intermediate,
    void* tmp,         // [N, intermediate]
    void* O,           // [N, hidden_dim]
    int dtype_code
);

torch::Tensor mlp_forward(//此处不再需要tmp作为输入
    torch::Tensor X,       // [N, hidden_dim]
    torch::Tensor gate,    // [intermediate, hidden_dim]
    torch::Tensor up,      // [intermediate, hidden_dim]
    torch::Tensor down    // [hidden_dim, intermediate]
) {
    // ---------------- CUDA checks ----------------
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(down.is_cuda(), "down must be a CUDA tensor");

    // ---------------- Device checks ----------------
    TORCH_CHECK(X.device() == gate.device(), "X/gate device mismatch");
    TORCH_CHECK(X.device() == up.device(), "X/up device mismatch");
    TORCH_CHECK(X.device() == down.device(), "X/down device mismatch");

    // ---------------- Shape checks ----------------
    TORCH_CHECK(X.dim() == 2, "X must be [N, hidden_dim]");
    TORCH_CHECK(gate.dim() == 2, "gate must be 2D");
    TORCH_CHECK(up.dim() == 2, "up must be 2D");
    TORCH_CHECK(down.dim() == 2, "down must be 2D");

    const int64_t N = X.size(0);
    const int64_t hidden_dim = X.size(1);
    const int64_t intermediate = gate.size(0);

    TORCH_CHECK(N > 0, "N must be positive");
    TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
    TORCH_CHECK(intermediate > 0, "intermediate must be positive");

    TORCH_CHECK(gate.size(1) == hidden_dim, "gate shape mismatch");
    TORCH_CHECK(up.size(0) == intermediate && up.size(1) == hidden_dim, "up shape mismatch");

    TORCH_CHECK(down.size(0) == hidden_dim && down.size(1) == intermediate, "down shape mismatch");

    // ---------------- Batch size warning ----------------
    if (N > 32) {
        TORCH_WARN_ONCE("MLP: N = ", N, " is relatively large, which may result in poor performance.");
    }
    TORCH_CHECK(N <= 65535, "N exceeds the supported gridDim.y limit (65535)");

    // ---------------- Dtype checks ----------------
    TORCH_CHECK(
        X.scalar_type() == gate.scalar_type() &&
        X.scalar_type() == up.scalar_type() &&
        X.scalar_type() == down.scalar_type(),
        "All tensors must have the same dtype"
    );

    int dtype_code = -1;
    if (X.scalar_type() == at::kHalf) {
        dtype_code = 0;
    }
    else if (X.scalar_type() == at::kBFloat16) {
        dtype_code = 1;
    }
    else if (X.scalar_type() == at::kFloat) {
        dtype_code = 2;
    }
    else {
        TORCH_CHECK(false, "MLP only supports FP16, BF16 and FP32");
    }

    // ---------------- Contiguous checks ----------------
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    TORCH_CHECK(down.is_contiguous(), "down must be contiguous");

    c10::cuda::CUDAGuard device_guard(X.device());
    auto O = torch::empty({N, hidden_dim}, X.options());
    auto tmp = torch::empty({N, intermediate}, X.options());// 临时 Buffer：由 PyTorch Caching Allocator 管理

    mlp_forward_cuda(
        X.data_ptr(),
        gate.data_ptr(),
        up.data_ptr(),
        down.data_ptr(),
        static_cast<size_t>(N),
        static_cast<size_t>(hidden_dim),
        static_cast<size_t>(intermediate),
        tmp.data_ptr(),
        O.data_ptr(),
        dtype_code
    );
    return O;
}







// 进行python函数注册
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

    m.def(
        "rmsnorm_forward",
        &rmsnorm_forward,
        "tinyInfer RMSNorm forward",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("eps")
    );

    m.def(
        "rope_forward",
        &rope_forward,
        "Custom RoPE CUDA",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("positions"),
        pybind11::arg("inv_freq")
    );

    m.def(
        "linear_forward_cutlass",
        &linear_forward_cutlass,
        "CUTLASS Linear Forward",
        pybind11::arg("X"),
        pybind11::arg("W")
    );

    m.def(
        "linear_forward_cublas",
        &linear_forward_cublas,
        "Cublas Linear Forward",
        pybind11::arg("X"),
        pybind11::arg("W")
    );

    m.def(
        "mlp_forward",
        &mlp_forward,
        "MLP Forward",
        pybind11::arg("X"),
        pybind11::arg("gate"),
        pybind11::arg("up"),
        pybind11::arg("down")
    );
}



