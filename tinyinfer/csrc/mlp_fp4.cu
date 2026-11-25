#include "utils.cuh"


constexpr int MLP_THREADS = 256;
constexpr int MLP_NUM_WARPS = MLP_THREADS / 32;


// v1: 与scale相乘放在循环内
// 针对fp16/bf16, VEC_SIZE = 8
/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage1_mxfp4(
    const T* __restrict__ X,     // [N, hidden_dim]
    const uint8_t* __restrict__ gate,  // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ gate_scale,  // [intermediate, hidden_dim // 32]
    const uint8_t* __restrict__ up,    // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ up_scale,    // [intermediate, hidden_dim // 32]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t row = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if (row >= intermediate) return;

    float sum_x_gate = float(0), sum_x_up = float(0);

    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = row * (hidden_dim / 2);
    size_t base_scale = row * (hidden_dim / 32);

    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const uint32_t* G4 = reinterpret_cast<const uint32_t*>(gate + base2);
    const uint8_t* Gs = gate_scale + base_scale;
    const uint32_t* U4 = reinterpret_cast<const uint32_t*>(up + base2);
    const uint8_t* Us = up_scale + base_scale;

    size_t num_vec = hidden_dim / VEC_SIZE;
    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx]; // 包含8个16b数据
        uint32_t gate_tmp = G4[vec_idx]; // 包含8个4b数据
        uint32_t up_tmp = U4[vec_idx];
        float gate_scale_tmp = decode_e8m0(Gs[vec_idx >> 2]); // 每4个uint32向量(32个FP4元素)共享一个Scale
        float up_scale_tmp = decode_e8m0(Us[vec_idx >> 2]);

        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);

            uint32_t g_code = (gate_tmp >> (j * 4)) & 0xF;
            uint32_t u_code = (up_tmp >> (j * 4)) & 0xF;

            float g_val = decode_e2m1(g_code) * gate_scale_tmp;
            float u_val = decode_e2m1(u_code) * up_scale_tmp;

            sum_x_gate = fmaf(x, g_val, sum_x_gate);
            sum_x_up = fmaf(x, u_val, sum_x_up);
        }
    }
    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        O[bidy * intermediate + row] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}*/




// v2: 与scale相乘放在循环外
template <typename T, int VEC_SIZE>
__global__ void mlp_stage1_mxfp4(
    const T* __restrict__ X,     // [N, hidden_dim]
    const uint8_t* __restrict__ gate,  // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ gate_scale,  // [intermediate, hidden_dim // 32]
    const uint8_t* __restrict__ up,    // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ up_scale,    // [intermediate, hidden_dim // 32]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t row = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if (row >= intermediate) return;

    float sum_x_gate = float(0), sum_x_up = float(0);

    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = row * (hidden_dim / 2);
    size_t base_scale = row * (hidden_dim / 32);
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const uint32_t* G4 = reinterpret_cast<const uint32_t*>(gate + base2);
    const uint8_t* Gs = gate_scale + base_scale;
    const uint32_t* U4 = reinterpret_cast<const uint32_t*>(up + base2);
    const uint8_t* Us = up_scale + base_scale;

    size_t num_vec = hidden_dim / VEC_SIZE;
    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx]; // 包含8个16b数据
        uint32_t gate_tmp = G4[vec_idx]; // 包含8个4b数据
        uint32_t up_tmp = U4[vec_idx];
        float gate_scale_tmp = decode_e8m0(Gs[vec_idx >> 2]); // 每4个uint32向量(32个FP4元素)共享一个Scale
        float up_scale_tmp = decode_e8m0(Us[vec_idx >> 2]);

        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        float partial_sum_x_gate = 0.0, partial_sum_x_up = 0.0;
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);

            uint32_t g_code = (gate_tmp >> (j * 4)) & 0xF;
            uint32_t u_code = (up_tmp >> (j * 4)) & 0xF;

            partial_sum_x_gate = fmaf(x, decode_e2m1(g_code), partial_sum_x_gate);
            partial_sum_x_up = fmaf(x, decode_e2m1(u_code), partial_sum_x_up);
        }

        sum_x_gate = fmaf(partial_sum_x_gate, gate_scale_tmp, sum_x_gate);
        sum_x_up = fmaf(partial_sum_x_up, up_scale_tmp, sum_x_up);
    }
    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        O[bidy * intermediate + row] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}



// v3: 加两级流水线(性能回退)
// PP_STAGE = 2; 假设hidden_dim是VEC_SIZE的整数倍
/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage1_mxfp4(
    const T* __restrict__ X,             // [N, hidden_dim]
    const uint8_t* __restrict__ gate,    // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ gate_scale, // [intermediate, hidden_dim // 32]
    const uint8_t* __restrict__ up,      // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ up_scale, // [intermediate, hidden_dim // 32]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O                    // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t row = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if (row >= intermediate) return;

    float sum_x_gate = float(0), sum_x_up = float(0);

    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = row * (hidden_dim / 2);
    size_t base_scale = row * (hidden_dim / 32);

    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const uint32_t* G4 = reinterpret_cast<const uint32_t*>(gate + base2);
    const uint8_t* Gs = gate_scale + base_scale;
    const uint32_t* U4 = reinterpret_cast<const uint32_t*>(up + base2);
    const uint8_t* Us = up_scale + base_scale;

    size_t num_vec = hidden_dim / VEC_SIZE;

    // prologue: PP_STAGE - 1 轮预取
    float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 x0 = zero;
    uint32_t g0 = 0, u0 = 0;
    uint8_t gs0 = 127, us0 = 127;

    size_t idx0 = laneid;
    if (idx0 < num_vec) {
        x0 = X4[idx0];
        g0 = G4[idx0];
        u0 = U4[idx0];
        gs0 = Gs[idx0 >> 2];
        us0 = Us[idx0 >> 2];
    }

    size_t num_iters = (num_vec + warpSize - 1) / warpSize;

    // Steady State
    for (size_t iter = 1; iter < num_iters; ++iter) {
        size_t idx = laneid + iter * warpSize;

        // Load next batch
        float4 x1 = zero;
        uint32_t g1 = 0, u1 = 0;
        uint8_t gs1 = 127, us1 = 127;

        if (idx < num_vec) {
            x1 = X4[idx];
            g1 = G4[idx];
            u1 = U4[idx];
            gs1 = Gs[idx >> 2];
            us1 = Us[idx >> 2];
        }

        // Compute current batch
        float gate_scale_tmp = decode_e8m0(gs0);
        float up_scale_tmp = decode_e8m0(us0);

        const T* xptr = reinterpret_cast<const T*>(&x0);
        float partial_sum_x_gate = 0.0f;
        float partial_sum_x_up = 0.0f;

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(xptr[j]);

            uint32_t g_code = (g0 >> (j * 4)) & 0xF;
            uint32_t u_code = (u0 >> (j * 4)) & 0xF;

            partial_sum_x_gate = fmaf(x, decode_e2m1(g_code), partial_sum_x_gate);
            partial_sum_x_up = fmaf(x, decode_e2m1(u_code), partial_sum_x_up);
        }

        sum_x_gate = fmaf(partial_sum_x_gate, gate_scale_tmp, sum_x_gate);
        sum_x_up = fmaf(partial_sum_x_up, up_scale_tmp, sum_x_up);

        // Rotate buffers
        x0 = x1;
        g0 = g1;
        u0 = u1;
        gs0 = gs1;
        us0 = us1;
    }

    // Epilogue
    float gate_scale_tmp = decode_e8m0(gs0);
    float up_scale_tmp = decode_e8m0(us0);

    const T* xptr = reinterpret_cast<const T*>(&x0);
    float partial_sum_x_gate = 0.0f;
    float partial_sum_x_up = 0.0f;

    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
        float x = to_float(xptr[j]);

        uint32_t g_code = (g0 >> (j * 4)) & 0xF;
        uint32_t u_code = (u0 >> (j * 4)) & 0xF;

        partial_sum_x_gate = fmaf(
            x, decode_e2m1(g_code), partial_sum_x_gate
        );
        partial_sum_x_up = fmaf(
            x, decode_e2m1(u_code), partial_sum_x_up
        );
    }

    sum_x_gate = fmaf(
        partial_sum_x_gate, gate_scale_tmp, sum_x_gate
    );
    sum_x_up = fmaf(
        partial_sum_x_up, up_scale_tmp, sum_x_up
    );

    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if (laneid == 0) {
        O[bidy * intermediate + row] =
            from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}*/






// v4: decode融合
/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage1_mxfp4(
    const T* __restrict__ X,     // [N, hidden_dim]
    const uint8_t* __restrict__ gate,  // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ gate_scale,  // [intermediate, hidden_dim // 32]
    const uint8_t* __restrict__ up,    // [intermediate, hidden_dim // 2]
    const uint8_t* __restrict__ up_scale,    // [intermediate, hidden_dim // 32]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t row = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if (row >= intermediate) return;

    float sum_x_gate = float(0), sum_x_up = float(0);

    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = row * (hidden_dim / 2);
    size_t base_scale = row * (hidden_dim / 32);

    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const uint32_t* G4 = reinterpret_cast<const uint32_t*>(gate + base2);
    const uint8_t* Gs = gate_scale + base_scale;
    const uint32_t* U4 = reinterpret_cast<const uint32_t*>(up + base2);
    const uint8_t* Us = up_scale + base_scale;

    size_t num_vec = hidden_dim / VEC_SIZE;
    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx]; // 包含8个16b数据
        uint32_t gate_tmp = G4[vec_idx]; // 包含8个4b数据
        uint32_t up_tmp = U4[vec_idx];
        uint8_t gate_scale_tmp = Gs[vec_idx >> 2]; // 每4个uint32向量(32个FP4元素)共享一个Scale
        uint8_t up_scale_tmp = Us[vec_idx >> 2];

        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);

            uint32_t g_code = (gate_tmp >> (j * 4)) & 0xF;
            uint32_t u_code = (up_tmp >> (j * 4)) & 0xF;

            float g_val = decode_mxfp4_fused(g_code, gate_scale_tmp);
            float u_val = decode_mxfp4_fused(u_code, up_scale_tmp);

            sum_x_gate = fmaf(x, g_val, sum_x_gate);
            sum_x_up = fmaf(x, u_val, sum_x_up);
        }
    }
    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        O[bidy * intermediate + row] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}*/








// 针对fp16/bf16, VEC_SIZE = 8
template <typename T, int VEC_SIZE>
__global__ void mlp_stage2_mxfp4(
    const T* __restrict__ X,             // [N, intermediate]
    const uint8_t* __restrict__ down,    // [hidden_dim, intermediate / 2]
    const uint8_t* __restrict__ down_scale, // [hidden_dim, intermediate / 32]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O                    // [N, hidden_dim]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_down = 0.0f;

    size_t out_idx = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if (out_idx >= hidden_dim) {
        return;
    }

    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = out_idx * (intermediate / 2);
    size_t base_scale = out_idx * (intermediate / 32);

    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const uint32_t* D4 = reinterpret_cast<const uint32_t*>(down + base2);
    const uint8_t* Ds = down_scale + base_scale;

    size_t num_vec = intermediate / VEC_SIZE;

    for (size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx];
        uint32_t down_tmp = D4[vec_idx];

        // 每32个FP4权重共享一个E8M0 Scale
        float down_scale_tmp = decode_e8m0(Ds[vec_idx >> 2]);

        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        float partial_sum_x_down = 0.0;

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);

            // 每个uint32包含8个FP4元素
            uint32_t d_code = (down_tmp >> (j * 4)) & 0xF;

            partial_sum_x_down = fmaf(x, decode_e2m1(d_code), partial_sum_x_down);
        }
        
        sum_x_down = fmaf(partial_sum_x_down, down_scale_tmp, sum_x_down);
    }

    sum_x_down = warp_reduce_sum<float>(sum_x_down);

    if (laneid == 0) {
        O[static_cast<size_t>(bidy) * hidden_dim + out_idx] = from_float<T>(sum_x_down);
    }
}





// 针对MXFP4 Weight-only, X/tmp/O仍为FP16/BF16
template <typename T, int VEC_SIZE>
void launch_mlp_mxfp4_typed(
    const T* X,                   // [N, hidden_dim]
    const uint8_t* gate,         // [intermediate, hidden_dim / 2]
    const uint8_t* gate_scale,   // [intermediate, hidden_dim / 32]
    const uint8_t* up,           // [intermediate, hidden_dim / 2]
    const uint8_t* up_scale,     // [intermediate, hidden_dim / 32]
    const uint8_t* down,         // [hidden_dim, intermediate / 2]
    const uint8_t* down_scale,   // [hidden_dim, intermediate / 32]
    size_t N,
    size_t hidden_dim,
    size_t intermediate,
    T* tmp,                      // [N, intermediate]
    T* O                         // [N, hidden_dim]
) {
    if (hidden_dim % 32 != 0 || intermediate % 32 != 0) {
        throw std::runtime_error("Invalid MXFP4 MLP dimensions");
    }

    const dim3 block_dim(MLP_THREADS);
    const dim3 grid_dim_stage1((intermediate + MLP_NUM_WARPS - 1) / MLP_NUM_WARPS, N);
    const dim3 grid_dim_stage2((hidden_dim + MLP_NUM_WARPS - 1) / MLP_NUM_WARPS, N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mlp_stage1_mxfp4<T, VEC_SIZE><<<grid_dim_stage1, block_dim, 0, stream>>>(X, gate, gate_scale, up, up_scale, hidden_dim, intermediate, tmp);
    CUDA_CHECK(cudaGetLastError());
    mlp_stage2_mxfp4<T, VEC_SIZE><<<grid_dim_stage2, block_dim, 0, stream>>>(tmp, down, down_scale, hidden_dim, intermediate, O);
    CUDA_CHECK(cudaGetLastError());
}




extern "C" void mlp_forward_mxfp4_cuda(
    const void* X,              // [N, hidden_dim]
    const void* gate,           // [intermediate, hidden_dim / 2]
    const void* gate_scale,     // [intermediate, hidden_dim / 32]
    const void* up,             // [intermediate, hidden_dim / 2]
    const void* up_scale,       // [intermediate, hidden_dim / 32]
    const void* down,           // [hidden_dim, intermediate / 2]
    const void* down_scale,     // [hidden_dim, intermediate / 32]
    size_t N,
    size_t hidden_dim,
    size_t intermediate,
    void* tmp,                  // [N, intermediate]
    void* O,                    // [N, hidden_dim]
    int dtype_code
) {
    if (dtype_code == 0) { // half
        launch_mlp_mxfp4_typed<half, 8>(
            static_cast<const half*>(X),
            static_cast<const uint8_t*>(gate),
            static_cast<const uint8_t*>(gate_scale),
            static_cast<const uint8_t*>(up),
            static_cast<const uint8_t*>(up_scale),
            static_cast<const uint8_t*>(down),
            static_cast<const uint8_t*>(down_scale),
            N,
            hidden_dim,
            intermediate,
            static_cast<half*>(tmp),
            static_cast<half*>(O)
        );
        return;
    }

    if (dtype_code == 1) { // bfloat16
        launch_mlp_mxfp4_typed<__nv_bfloat16, 8>(
            static_cast<const __nv_bfloat16*>(X),
            static_cast<const uint8_t*>(gate),
            static_cast<const uint8_t*>(gate_scale),
            static_cast<const uint8_t*>(up),
            static_cast<const uint8_t*>(up_scale),
            static_cast<const uint8_t*>(down),
            static_cast<const uint8_t*>(down_scale),
            N,
            hidden_dim,
            intermediate,
            static_cast<__nv_bfloat16*>(tmp),
            static_cast<__nv_bfloat16*>(O)
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code for MXFP4 MLP");
}



