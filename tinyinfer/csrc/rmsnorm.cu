#include "utils.h"

constexpr int RMSNORM_NUM_WARPS = 4;
constexpr int RMSNORM_THREADS = 128;


// 对输入矩阵求rmsnorm;
// 并行策略: 每个block负责一个head的一行(hidden_dim); 每个block固定为128个thread/4warp
template <typename T>
__global__ void rmsnorm(
    const T* input,     //shape: [num_tokens, num_heads, hidden_dim] or [num_tokens, hidden_dim]
    size_t hidden_dim,
    const T* weight,    // shape: [hidden_dim]
    float eps, //直接用fp32
    T* output
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    float sum = float(0);//用fp32保护精度
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[bid * hidden_dim + i]);
        sum += x * x;
    }
    sum = warp_reduce_sum<float>(sum);//一级规约

    __shared__ float sum_warp[RMSNORM_NUM_WARPS];//准备二级规约数据
    __shared__ float inv_rms;//由该block所有thread共享访问, 因此存在smem中
    if(lane_id == 0){
        sum_warp[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < RMSNORM_NUM_WARPS) {//二级规约, 只需要lane0-3参与即可
        float val = sum_warp[lane_id];
        val += __shfl_xor_sync(0xF, val, 2, 4);
        val += __shfl_xor_sync(0xF, val, 1, 4);
        if (lane_id == 0) {
            inv_rms = rsqrtf(val / static_cast<float>(hidden_dim) + eps);
        }
    }
    __syncthreads();

    // write back
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[bid * hidden_dim + i]);//转为fp32保护精度
        float w = to_float(weight[i]);//fp32保护精度
        output[bid * hidden_dim + i] = from_float<T>(w * x * inv_rms);
    }
}




template <typename T>
void launch_rmsnorm_typed(
    const T* input,
    size_t hidden_dim,
    size_t num_rows,
    const T* weight,
    float eps,
    T* output
) {
    const dim3 block_dim(RMSNORM_THREADS);
    const dim3 grid_dim(num_rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm<T><<<grid_dim, block_dim, 0, stream>>>(input, hidden_dim, weight, eps, output);

    CUDA_CHECK(cudaGetLastError());
}




// 在python环境下, 由于dtype是运行时决定的而非编译器决定, 因此这里无法使用模板参数, 而只能动态传参dtype_code
extern "C" void rmsnorm_forward_cuda(
    const void* input,
    size_t hidden_dim,
    size_t num_rows,
    const void* weight,
    float eps,
    void* output,
    int dtype_code
) {
    if (dtype_code == 0) {
        launch_rmsnorm_typed<half>(
            static_cast<const half*>(input),
            hidden_dim,
            num_rows,
            static_cast<const half*>(weight),
            eps,
            static_cast<half*>(output)
        );
        return;
    }
    if (dtype_code == 1) {
        launch_rmsnorm_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(input),
            hidden_dim,
            num_rows,
            static_cast<const __nv_bfloat16*>(weight),
            eps,
            static_cast<__nv_bfloat16*>(output)
        );
        return;
    }
    if (dtype_code == 2) {
        launch_rmsnorm_typed<float>(
            static_cast<const float*>(input),
            hidden_dim,
            num_rows,
            static_cast<const float*>(weight),
            eps,
            static_cast<float*>(output)
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code");
}
