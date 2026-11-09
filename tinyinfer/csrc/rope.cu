#include "utils.h"

constexpr int ROPE_THREADS = 128;


// rope in half-split mode
// 每个block负责D个元素; 总block数目 = max(THq, THk)
// 需要确保RD为偶数; q/k/freq均contiguous
//inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
template <typename T>
__global__ void rope(
    const T* __restrict__ q,         // [T, Hq, D]
    const T* __restrict__ k,         // [T, Hk, D]
    const int64_t* __restrict__ positions, // [T]
    const float* __restrict__ inv_freq, // [RD/2]
    int hidden_dim,     // D
    int rotary_dim,     // RD
    int num_tokens,     // T
    int head_q,         // Hq
    int head_k,         // Hk
    T* __restrict__ q_rot,           // [T, Hq, D]
    T* __restrict__ k_rot            // [T, Hk, D]
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int token_id_q = bid / head_q;
    int token_id_k = bid / head_k;
    int half = rotary_dim / 2;

    // q rotate
    if(token_id_q < num_tokens) {
        for(int i = tid; i < hidden_dim; i += blockDim.x){
            float result;
            int64_t index = static_cast<int64_t>(bid) * hidden_dim + i;
            if(i >= rotary_dim) {//尾部不需要rotate的维度
                result = to_float(q[index]);
            }
            else if(i < half) {//first half
                float theta = static_cast<float>(positions[token_id_q]) * inv_freq[i];
                float cos, sin; sincosf(theta, &sin, &cos);
                result = to_float(q[index]) * cos - to_float(q[index + half]) * sin;
            }
            else {//second half
                float theta = static_cast<float>(positions[token_id_q]) * inv_freq[i - half];
                float cos, sin; sincosf(theta, &sin, &cos);
                result = to_float(q[index]) * cos + to_float(q[index - half]) * sin;
            }
            q_rot[index] = from_float<T>(result);
        }
    }


    // k rotate
    if(token_id_k < num_tokens) {
        for(int i = tid; i < hidden_dim; i += blockDim.x) {
            float result;
            int64_t index = static_cast<int64_t>(bid) * hidden_dim + i;
            if(i >= rotary_dim) {
                result = to_float(k[index]);
            }
            else if(i < rotary_dim / 2) {
                float theta = static_cast<float>(positions[token_id_k]) * inv_freq[i];
                float cos, sin; sincosf(theta, &sin, &cos);
                result = to_float(k[index]) * cos - to_float(k[index + half]) * sin;
            }
            else {
                float theta = static_cast<float>(positions[token_id_k]) * inv_freq[i - half];
                float cos, sin; sincosf(theta, &sin, &cos);
                result = to_float(k[index]) * cos + to_float(k[index - half]) * sin;
            }
            k_rot[index] = from_float<T>(result);
        }
    }

}





template <typename T>
void launch_rope_typed(
    const T* q,         // [T, Hq, D]
    const T* k,         // [T, Hk, D]
    const int64_t* positions, // [T]
    const float* inv_freq, // [RD/2]
    int hidden_dim,     // D
    int rotary_dim,     // RD
    int num_tokens,     // T
    int head_q,         // Hq
    int head_k,         // Hk
    T* q_rot,           // [T, Hq, D]
    T* k_rot            // [T, Hk, D]
    
) {
    const dim3 block_dim(ROPE_THREADS);
    int64_t q_blocks = static_cast<int64_t>(num_tokens) * head_q;
    int64_t k_blocks = static_cast<int64_t>(num_tokens) * head_k;
    int64_t total_blocks = std::max(q_blocks, k_blocks);
    dim3 grid_dim(total_blocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope<T><<<grid_dim, block_dim, 0, stream>>>(q, k, positions, inv_freq, hidden_dim, rotary_dim, num_tokens, head_q, head_k, q_rot, k_rot);

    CUDA_CHECK(cudaGetLastError());
}






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
) {
    if (dtype_code == 0) {
        launch_rope_typed<half>(
            static_cast<const half*>(q),
            static_cast<const half*>(k),
            positions,
            inv_freq,
            hidden_dim,
            rotary_dim,
            num_tokens,
            head_q,
            head_k,
            static_cast<half*>(q_rot),
            static_cast<half*>(k_rot)
        );
        return;
    }
    if (dtype_code == 1) {
        launch_rope_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k),
            positions,
            inv_freq,
            hidden_dim,
            rotary_dim,
            num_tokens,
            head_q,
            head_k,
            static_cast<__nv_bfloat16*>(q_rot),
            static_cast<__nv_bfloat16*>(k_rot)
        );
        return;
    }
    if (dtype_code == 2) {
        launch_rope_typed<float>(
            static_cast<const float*>(q),
            static_cast<const float*>(k),
            positions,
            inv_freq,
            hidden_dim,
            rotary_dim,
            num_tokens,
            head_q,
            head_k,
            static_cast<float*>(q_rot),
            static_cast<float*>(k_rot)
        );
        return;
    }

    throw std::runtime_error("unsupported dtype_code");
}
