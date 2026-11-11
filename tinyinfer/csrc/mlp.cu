#include "utils.h"


constexpr int MLP_THREADS = 256;
constexpr int MLP_NUM_WARPS = 8;



// 实现qwen3中的mlp layer, 假设seq_len = small N;
// 并行策略: 一个block负责一组向量乘累加

// stage1: O = silu(x @ gate) * (x @ up); gridDim = [intermediate, N]
/*template <typename T>
__global__ void mlp_stage1(
    const T* __restrict__ X,     // [N, hidden_dim]
    const T* __restrict__ gate,  // [intermediate, hidden_dim]
    const T* __restrict__ up,    // [intermediate, hidden_dim]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, intermediate]
) {
    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    __shared__ float tmp_sum_x_gate[MLP_NUM_WARPS]; //中间值用fp32保留精度
    __shared__ float tmp_sum_x_up[MLP_NUM_WARPS];

    float sum_x_gate = float(0), sum_x_up = float(0);
    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = static_cast<size_t>(bidx) * hidden_dim;
    for(int i = tid; i < hidden_dim; i += MLP_THREADS) {
        float x = to_float(X[base1 + i]);
        sum_x_gate = fmaf(x, to_float(gate[base2 + i]), sum_x_gate);
        sum_x_up = fmaf(x, to_float(up[base2 + i]), sum_x_up);
    }
    sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 一级规约
    sum_x_up = warp_reduce_sum<float>(sum_x_up);

    if(laneid == 0) {
        tmp_sum_x_gate[warpid] = sum_x_gate;
        tmp_sum_x_up[warpid] = sum_x_up;
    }
    __syncthreads();

    if(warpid == 0) {
        sum_x_gate = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_gate[laneid] : float(0);
        sum_x_up = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_up[laneid] : float(0);
        sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 二级规约
        sum_x_up = warp_reduce_sum<float>(sum_x_up);
        if(laneid == 0) {
            O[bidy * intermediate + bidx] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
        }
    }
}*/


// stage1: O = silu(x @ gate) * (x @ up); gridDim = [intermediate, N]
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

    __shared__ float tmp_sum_x_gate[MLP_NUM_WARPS]; //中间值用fp32保留精度
    __shared__ float tmp_sum_x_up[MLP_NUM_WARPS];

    float sum_x_gate = float(0), sum_x_up = float(0);
    size_t base1 = static_cast<size_t>(bidy) * hidden_dim;
    size_t base2 = static_cast<size_t>(bidx) * hidden_dim;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* G4 = reinterpret_cast<const float4*>(gate + base2);
    const float4* U4 = reinterpret_cast<const float4*>(up + base2);
    size_t num_vec = hidden_dim / VEC_SIZE;

    for(size_t vec_idx = tid; vec_idx < num_vec; vec_idx += MLP_THREADS) {
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
        tmp_sum_x_gate[warpid] = sum_x_gate;
        tmp_sum_x_up[warpid] = sum_x_up;
    }
    __syncthreads();

    if(warpid == 0) {
        sum_x_gate = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_gate[laneid] : float(0);
        sum_x_up = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_up[laneid] : float(0);
        sum_x_gate = warp_reduce_sum<float>(sum_x_gate); // 二级规约
        sum_x_up = warp_reduce_sum<float>(sum_x_up);
        if(laneid == 0) {
            O[bidy * intermediate + bidx] = from_float<T>(silu_f(sum_x_gate) * sum_x_up);
        }
    }
}


// stage2: O = x @ down; gridDim = [hidden_dim, N]
/*template <typename T>
__global__ void mlp_stage2(
    const T* __restrict__ X,     // [N, intermediate]
    const T* __restrict__ down,  // [hidden_dim, intermediate]
    size_t hidden_dim,
    size_t intermediate,
    T* __restrict__ O            // [N, hidden_dim]
) {
    int tid = threadIdx.x;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    __shared__ float tmp_sum_x_down[MLP_NUM_WARPS]; //中间值用fp32保留精度

    float sum_x_down = float(0);
    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = static_cast<size_t>(bidx) * intermediate;
    for(int i = tid; i < intermediate; i += MLP_THREADS) {
        sum_x_down = fmaf(to_float(X[base1 + i]), to_float(down[base2 + i]), sum_x_down);
    }
    sum_x_down = warp_reduce_sum<float>(sum_x_down);
    
    if(laneid == 0) {
        tmp_sum_x_down[warpid] = sum_x_down;
    }
    __syncthreads();

    if(warpid == 0) {
        sum_x_down = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_down[laneid] : float(0);
        sum_x_down = warp_reduce_sum(sum_x_down);
        if(laneid == 0) {
            O[bidy * hidden_dim + bidx] = from_float<T>(sum_x_down);
        }
    }
}*/


// stage2: O = x @ down; gridDim = [hidden_dim, N]
template <typename T, int VEC_SIZE>
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

    __shared__ float tmp_sum_x_down[MLP_NUM_WARPS]; //中间值用fp32保留精度

    float sum_x_down = float(0);
    size_t base1 = static_cast<size_t>(bidy) * intermediate;
    size_t base2 = static_cast<size_t>(bidx) * intermediate;
    const float4* X4 = reinterpret_cast<const float4*>(X + base1);
    const float4* D4 = reinterpret_cast<const float4*>(down + base2);
    size_t num_vec = intermediate / VEC_SIZE;

    for(size_t vec_idx = tid; vec_idx < num_vec; vec_idx += MLP_THREADS) {
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
        tmp_sum_x_down[warpid] = sum_x_down;
    }
    __syncthreads();

    if(warpid == 0) {
        sum_x_down = (laneid < MLP_NUM_WARPS) ? tmp_sum_x_down[laneid] : float(0);
        sum_x_down = warp_reduce_sum(sum_x_down);
        if(laneid == 0) {
            O[bidy * hidden_dim + bidx] = from_float<T>(sum_x_down);
        }
    }
}







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
    const dim3 block_dim(MLP_THREADS);
    const dim3 grid_dim_stage1(intermediate, N);
    const dim3 grid_dim_stage2(hidden_dim, N);

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
    if (dtype_code == 0) {
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