#include "utils.cuh"

constexpr int LM_HEAD_THREADS = 128;
constexpr int LM_HEAD_NUM_WARPS = LM_HEAD_THREADS / 32;

/************************ gemm_ccore_mxfp4 ************************/
// 针对fp16/bf16, VEC_SIZE = 8
template <typename T, int VEC_SIZE, int M_TILE>
__global__ void lm_head_gemm_ccore_mxfp4(
    const T* __restrict__ X,                  // [M, hidden_size]
    const uint8_t* __restrict__ weight,       // [vocab_size, hidden_size / 2]
    const uint8_t* __restrict__ weight_scale, // [vocab_size, hidden_size / 32]
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    T* __restrict__ O                        // [M, vocab_size]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t vocab_idx = static_cast<size_t>(bidx) * LM_HEAD_NUM_WARPS + warpid;
    if (vocab_idx >= vocab_size) return;

    size_t base_w = vocab_idx * (hidden_size / 2);
    size_t base_scale = vocab_idx * (hidden_size / 32);

    const uint32_t* W4 = reinterpret_cast<const uint32_t*>(weight + base_w);
    const uint8_t* Ws = weight_scale + base_scale;

    size_t num_vec = hidden_size / VEC_SIZE;

    float sum[M_TILE] = {0.f};

    for (size_t idx = laneid; idx < num_vec; idx += warpSize) {
        uint32_t w = W4[idx];

        // 每32个FP4权重共享一个E8M0 Scale
        float w_scale = decode_e8m0(Ws[idx >> 2]);

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            size_t token_idx = static_cast<size_t>(bidy) * M_TILE + m;
            if (token_idx < M) {
                const float4* X4 = reinterpret_cast<const float4*>(X + token_idx * hidden_size);
                float4 x_tmp = X4[idx];

                const T* xptr = reinterpret_cast<const T*>(&x_tmp);
                float partial_sum = 0.f;

                #pragma unroll
                for (int j = 0; j < VEC_SIZE; ++j) {
                    float x = to_float(xptr[j]);

                    // 每个uint32包含8个FP4元素
                    uint32_t w_code = (w >> (j * 4)) & 0xF;

                    partial_sum = fmaf(x, decode_e2m1(w_code), partial_sum);
                }

                sum[m] = fmaf(partial_sum, w_scale, sum[m]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        size_t token_idx = static_cast<size_t>(bidy) * M_TILE + m;
        if (token_idx < M) {
            float result = warp_reduce_sum<float>(sum[m]);
            if (laneid == 0) {
                O[token_idx * vocab_size + vocab_idx] = from_float<T>(result);
            }
        }
    }
}



template <typename T, int VEC_SIZE, int M_TILE>
void launch_lm_head_gemm_ccore_mxfp4_typed(
    const T* X,
    const uint8_t* weight,
    const uint8_t* weight_scale, 
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    T* O
) {
    const dim3 block_dim(LM_HEAD_THREADS);
    const dim3 grid_dim((vocab_size + LM_HEAD_NUM_WARPS - 1) / LM_HEAD_NUM_WARPS, (M + M_TILE - 1) / M_TILE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lm_head_gemm_ccore_mxfp4<T, VEC_SIZE, M_TILE><<<grid_dim, block_dim, 0, stream>>>(X, weight, weight_scale, M, hidden_size, vocab_size, O);

    CUDA_CHECK(cudaGetLastError());
}





extern "C" void lm_head_forward_mxfp4_cuda(
    const void* X,              // [M, hidden_size]
    const void* weight,         // [vocab_size, hidden_size / 2]
    const void* weight_scale,   // [vocab_size, hidden_size / 32]
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    void* O,
    int dtype_code
) {
    if (dtype_code == 0) { // half
        if(M <= 2) {
            launch_lm_head_gemm_ccore_mxfp4_typed<half, 8, 2>(
                static_cast<const half*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<half*>(O)
            );
        }
        else if(M <= 4)  {
            launch_lm_head_gemm_ccore_mxfp4_typed<half, 8, 4>(
                static_cast<const half*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<half*>(O)
            );
        }
        else if(M <= 6) {
            launch_lm_head_gemm_ccore_mxfp4_typed<half, 8, 8>(
                static_cast<const half*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<half*>(O)
            );
        }
        else {
            // TODO: launch_lm_head_gemm_tcore_mxfp4_typed
            launch_lm_head_gemm_ccore_mxfp4_typed<half, 8, 8>(
                static_cast<const half*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<half*>(O)
            );
        }
        return;
    }

    if (dtype_code == 1) { // bf16
        if(M <= 2) {
            launch_lm_head_gemm_ccore_mxfp4_typed<__nv_bfloat16, 8, 2>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<__nv_bfloat16*>(O)
            );
        }
        else if(M <= 4)  {
            launch_lm_head_gemm_ccore_mxfp4_typed<__nv_bfloat16, 8, 4>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<__nv_bfloat16*>(O)
            );
        }
        else if(M <= 6) {
            launch_lm_head_gemm_ccore_mxfp4_typed<__nv_bfloat16, 8, 8>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<__nv_bfloat16*>(O)
            );
        }
        else {
            // TODO: launch_lm_head_gemm_tcore_mxfp4_typed
            launch_lm_head_gemm_ccore_mxfp4_typed<__nv_bfloat16, 8, 8>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint8_t*>(weight),
                static_cast<const uint8_t*>(weight_scale),
                M,
                hidden_size,
                vocab_size,
                static_cast<__nv_bfloat16*>(O)
            );
        }
        return;
    }


    throw std::runtime_error("unsupported dtype_code for MXFP4 LM_head");
}