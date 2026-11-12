#include "utils.h"


constexpr int MLP_THREADS = 256;
constexpr int MLP_NUM_WARPS = MLP_THREADS / 32;



// 实现qwen3中的mlp layer, 假设seq_len = small N;
// 并行策略: 一个block负责一组向量乘累加

// stage1: O = silu(x @ gate) * (x @ up); gridDim = [intermediate, N]
/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage1(
    const T* __restrict__ X,     // [N, hidden_dim]
    const T* __restrict__ gate,  // [intermediate, hidden_dim]
    const T* __restrict__ up,    // [intermediate, hidden_dim]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_gate = float(0), sum_x_up = float(0);
    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = (static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid) * hidden_dim;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* G4 = reinterpret_cast<const float4*>(gate + base2);
    const float4* U4 = reinterpret_cast<const float4*>(up + base2);
    size_t num_vec = hidden_dim / VEC_SIZE;

    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx];
        float4 gate_tmp = G4[vec_idx];
        float4 up_tmp = U4[vec_idx];
        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        const T* gate_ptr = reinterpret_cast<const T*>(&gate_tmp);
        const T* up_ptr = reinterpret_cast<const T*>(&up_tmp);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);
            sum_x_gate = fmaf(x, to_float(gate_ptr[j]), sum_x_gate);
            sum_x_up = fmaf(x, to_float(up_ptr[j]), sum_x_up);
        }
    }
    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        O[bidy * intermediate + (bidx * MLP_NUM_WARPS + warpid)] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}*/



// PP_STAGE = 2; 假设hidden_dim是VEC_SIZE的整数倍
template <typename T, int VEC_SIZE>
__global__ void mlp_stage1(
    const T* __restrict__ X,     // [N, hidden_dim]
    const T* __restrict__ gate,  // [intermediate, hidden_dim]
    const T* __restrict__ up,    // [intermediate, hidden_dim]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_gate = float(0), sum_x_up = float(0);
    size_t out_idx = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if(out_idx >= intermediate) {
        return;
    }
    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = out_idx * hidden_dim;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* G4 = reinterpret_cast<const float4*>(gate + base2);
    const float4* U4 = reinterpret_cast<const float4*>(up + base2);
    size_t num_vec = hidden_dim / VEC_SIZE;

    
    // prologue: PP_STAGE - 1 轮预取
    float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 x0 = zero, g0 = zero, u0 = zero;
    size_t idx0 = laneid;
    if(idx0 < num_vec) {
        x0 = X4[idx0]; g0 = G4[idx0]; u0 = U4[idx0];
    }
    size_t num_iters = (num_vec + warpSize - 1) / warpSize;

    // Steady State
    for(size_t iter = 1; iter < num_iters; ++iter) {
        size_t idx = laneid + iter * warpSize;

        // Load next batch
        float4 x1 = zero, g1 = zero, u1 = zero;
        if(idx < num_vec) {
            x1 = X4[idx]; g1 = G4[idx]; u1 = U4[idx];
        }

        // Compute current batch
        const T* xptr = reinterpret_cast<const T*>(&x0);
        const T* gptr = reinterpret_cast<const T*>(&g0);
        const T* uptr = reinterpret_cast<const T*>(&u0);
        #pragma unroll
        for(int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(xptr[j]);
            sum_x_gate = fmaf(x, to_float(gptr[j]), sum_x_gate);
            sum_x_up = fmaf(x, to_float(uptr[j]), sum_x_up);
        }

        // Rotate buffers
        x0 = x1; g0 = g1; u0 = u1;
    }

    // Epilogue
    const T* xptr = reinterpret_cast<const T*>(&x0);
    const T* gptr = reinterpret_cast<const T*>(&g0);
    const T* uptr = reinterpret_cast<const T*>(&u0);
    #pragma unroll
    for(int j = 0; j < VEC_SIZE; ++j) {
        float x = to_float(xptr[j]);
        sum_x_gate = fmaf(x, to_float(gptr[j]), sum_x_gate);
        sum_x_up = fmaf(x, to_float(uptr[j]), sum_x_up);
    }

    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        O[bidy * intermediate + out_idx] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
    }
}




// stage2: O = x @ down; gridDim = [hidden_dim, N]
/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage2(
    const T* __restrict__ X,     // [N, intermediate]
    const T* __restrict__ down,  // [hidden_dim, intermediate]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, hidden_dim]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_down = float(0);
    size_t out_idx = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if(out_idx >= hidden_dim) {
        return; 
    }
    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = out_idx * intermediate;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* D4 = reinterpret_cast<const float4*>(down + base2);
    size_t num_vec = intermediate / VEC_SIZE;

    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx];
        float4 down_tmp = D4[vec_idx];
        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        const T* down_ptr = reinterpret_cast<const T*>(&down_tmp);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            sum_x_down = fmaf(to_float(x_ptr[j]), to_float(down_ptr[j]), sum_x_down);
        }
    }
    sum_x_down = warp_reduce_sum<float>(sum_x_down);
    
    if(laneid == 0) {
        O[bidy * hidden_dim + out_idx] = from_float<T>(sum_x_down);
    }
}*/



// PP_STAGE = 2; 假设intermediate是VEC_SIZE的整数倍
template <typename T, int VEC_SIZE>
__global__ void mlp_stage2(
    const T* __restrict__ X,     // [N, intermediate]
    const T* __restrict__ down,  // [hidden_dim, intermediate]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O           // [N, hidden_dim]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_down = float(0);
    size_t out_idx = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if(out_idx >= hidden_dim) {
        return;
    }
    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = out_idx * intermediate;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* D4 = reinterpret_cast<const float4*>(down + base2);
    size_t num_vec = intermediate / VEC_SIZE;


    // prologue: PP_STAGE - 1 轮预取
    float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 x0 = zero, d0 = zero;
    size_t idx0 = laneid;
    if(idx0 < num_vec) {
        x0 = X4[idx0]; d0 = D4[idx0];
    }
    size_t num_iters = (num_vec + warpSize - 1) / warpSize;

    // Steady State
    for(size_t iter = 1; iter < num_iters; ++iter) {
        size_t idx = laneid + iter * warpSize;

        // Load next batch
        float4 x1 = zero, d1 = zero;
        if(idx < num_vec) {
            x1 = X4[idx]; d1 = D4[idx];
        }

        // Compute current batch
        const T* xptr = reinterpret_cast<const T*>(&x0);
        const T* dptr = reinterpret_cast<const T*>(&d0);
        #pragma unroll
        for(int j = 0; j < VEC_SIZE; ++j) {
            sum_x_down = fmaf(to_float(xptr[j]), to_float(dptr[j]), sum_x_down);
        }

        // Rotate buffers
        x0 = x1; d0 = d1;
    }

    // Epilogue
    const T* xptr = reinterpret_cast<const T*>(&x0);
    const T* dptr = reinterpret_cast<const T*>(&d0);
    #pragma unroll
    for(int j = 0; j < VEC_SIZE; ++j) {
        sum_x_down = fmaf(to_float(xptr[j]), to_float(dptr[j]), sum_x_down);
    }

    sum_x_down = warp_reduce_sum<float>(sum_x_down);

    if(laneid == 0) {
        O[bidy * hidden_dim + out_idx] = from_float<T>(sum_x_down);
    }
}






/*template <typename T, int VEC_SIZE>
__global__ void mlp_stage2(
    const T* __restrict__ X,     // [N, intermediate]
    const T* __restrict__ down,  // [hidden_dim, intermediate]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, hidden_dim]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum_x_down = float(0);
    float acc0 = 0.f;
    float acc1 = 0.f;
    float acc2 = 0.f;
    float acc3 = 0.f;
    size_t out_idx = static_cast<size_t>(bidx) * MLP_NUM_WARPS + warpid;
    if(out_idx >= hidden_dim) {
        return; 
    }
    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = out_idx * intermediate;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* D4 = reinterpret_cast<const float4*>(down + base2);
    size_t num_vec = intermediate / VEC_SIZE;

    for(size_t vec_idx = laneid; vec_idx < num_vec; vec_idx += warpSize) {
        float4 x_tmp = X4[vec_idx];
        float4 down_tmp = D4[vec_idx];
        const T* x_ptr = reinterpret_cast<const T*>(&x_tmp);
        const T* down_ptr = reinterpret_cast<const T*>(&down_tmp);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            float x = to_float(x_ptr[j]);
            float w = to_float(down_ptr[j]);
            switch (j % 4) {
                case 0: acc0 = fmaf(x, w, acc0); break;
                case 1: acc1 = fmaf(x, w, acc1); break;
                case 2: acc2 = fmaf(x, w, acc2); break;
                case 3: acc3 = fmaf(x, w, acc3); break;
            }
        }
    }
    sum_x_down = (acc0 + acc1) + (acc2 + acc3);
    sum_x_down = warp_reduce_sum<float>(sum_x_down);
    
    if(laneid == 0) {
        O[bidy * hidden_dim + out_idx] = from_float<T>(sum_x_down);
    }
}*/






template <typename T, int VEC_SIZE>
void launch_mlp_typed(
    const T* X,     // [N, hidden_dim]
    const T* gate,  // [intermediate, hidden_dim]
    const T* up,    // [intermediate, hidden_dim]
    const T* down,  // [hidden_dim, intermediate]
    size_t N,
    size_t hidden_dim,
    size_t intermediate,
    T* tmp,         // [N, intermediate]
    T* O            // [N, hidden_dim]
) {
    if (hidden_dim % VEC_SIZE != 0 || intermediate % VEC_SIZE != 0) {//启用向量加载需要满足尺寸为VEC_SIZE的整数倍
        throw std::runtime_error("Invalid MLP dimensions");
    }
    const dim3 block_dim(MLP_THREADS);
    const dim3 grid_dim_stage1((intermediate + MLP_NUM_WARPS - 1) / MLP_NUM_WARPS , N);
    const dim3 grid_dim_stage2((hidden_dim + MLP_NUM_WARPS - 1) / MLP_NUM_WARPS, N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mlp_stage1<T, VEC_SIZE><<<grid_dim_stage1, block_dim, 0, stream>>>(X, gate, up, hidden_dim, intermediate, tmp);
    CUDA_CHECK(cudaGetLastError());
    mlp_stage2<T, VEC_SIZE><<<grid_dim_stage2, block_dim, 0, stream>>>(tmp, down, hidden_dim, intermediate, O);
    CUDA_CHECK(cudaGetLastError());
}







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
) {
    if (dtype_code == 0) {// half
        launch_mlp_typed<half, 8>(
            static_cast<const half*>(X),
            static_cast<const half*>(gate),
            static_cast<const half*>(up),
            static_cast<const half*>(down),
            N,
            hidden_dim,
            intermediate,
            static_cast<half*>(tmp),
            static_cast<half*>(O)
        );
        return;
    }
    if (dtype_code == 1) {
        launch_mlp_typed<__nv_bfloat16, 8>(
            static_cast<const __nv_bfloat16*>(X),
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(up),
            static_cast<const __nv_bfloat16*>(down),
            N, 
            hidden_dim,
            intermediate,
            static_cast<__nv_bfloat16*>(tmp),
            static_cast<__nv_bfloat16*>(O)
        );
        return;
    }
    if (dtype_code == 2) {
        launch_mlp_typed<float, 4>(
            static_cast<const float*>(X),
            static_cast<const float*>(gate),
            static_cast<const float*>(up),
            static_cast<const float*>(down),
            N, 
            hidden_dim,
            intermediate,
            static_cast<float*>(tmp),
            static_cast<float*>(O)
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code");
}