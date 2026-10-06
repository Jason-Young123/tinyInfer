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



// 注意bf16到float之间的数值转换不支持float()/bf16(), 因此必须要依据static_cast做类型转换helper
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






//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flash_attention( 
    int target_seq_len, 
    int src_seq_len, 
    int q_heads, 
    int kv_heads, 
    int head_dim, 
    // 当前 Q[0] 在完整 K/V 序列中的绝对 token 位置。
    // 普通 prefill:      query_start_pos = 0
    // chunked prefill:   query_start_pos = 已计算 token 数
    // decode:            query_start_pos = 当前 decode token 的位置
    int query_start_pos,
    float scale,
    bool is_causal, 

    const T* Q, 
    const T* K, 
    const T* V, 
    T* O
) {
    int tid_x = threadIdx.x;//横向,blockDim.x列
    int tid_y = threadIdx.y;//纵向,blockDim.y行
    int bid_x = blockIdx.x;//x方向,总数 = #q_heads
    int bid_y = blockIdx.y;//y方向,总数 = #batch
    int bid_z = blockIdx.z;//z方向,总数 = Tr
    const int p = q_heads / kv_heads;//计算比例系数
    const int Br = blockDim.y;//Q纵向每块大小, 默认为32 (RTX 5090)
    const int Bc = blockDim.x;//K/V纵向分块大小, 默认为32
    const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

    //预计算常量; 为了和SDPA同步, scale已经作为入口参数, 而不需要在内部计算
    //const double scale_factor = 1.0 / sqrt(double(head_dim));//保留精度,采用double
    const double scale_factor = double(scale);

    //定义一系列临时变量
    extern __shared__ char shared_mem[];
    char* ptr = shared_mem;  
    //计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new; 为保留精度, SP采用double
    double* SP = reinterpret_cast<double*>(ptr);    // double SP[Br][Bc]
    ptr += Br * Bc * sizeof(double);
    float* m_prev = reinterpret_cast<float*>(ptr);  // float m_prev[Br]
    ptr += Br * sizeof(float);
    float* m_new = reinterpret_cast<float*>(ptr);   // float m_new[Br] 
    ptr += Br * sizeof(float);
    float* l_prev = reinterpret_cast<float*>(ptr);  // float l_prev[Br]
    ptr += Br * sizeof(float);
    float* l_new = reinterpret_cast<float*>(ptr);   // float l_new[Br] 
    ptr += Br * sizeof(float);  

    //原始数据QKV和计算结果O; 全采用float
    float* Q_sm = reinterpret_cast<float*>(ptr);    // float Q_sm[Br][head_dim] 
    ptr += Br * head_dim * sizeof(float);  
    float* K_T_sm = reinterpret_cast<float*>(ptr);  // float K_T_sm[head_dim][Bc]
    ptr += head_dim * Bc * sizeof(float);
    float* V_sm = reinterpret_cast<float*>(ptr);    // float V_sm[Br][head_dim] 
    ptr += Bc * head_dim * sizeof(float);  
    float* O_sm = reinterpret_cast<float*>(ptr);    // float O_sm[Br][head_dim]

    //定义访问宏
    #define   SP_AT(y, x)       SP[y * Bc + x]
    #define   Q_sm_AT(y, x)     Q_sm[y * head_dim + x]
    #define   K_T_sm_AT(y, x)   K_T_sm[y * Bc + x]
    #define   V_sm_AT(y, x)     V_sm[y * head_dim + x]
    #define   O_sm_AT(y, x)     O_sm[y * head_dim + x]


    /****************************preparation**************************/
    int bound_tid_y = ::min(Br, target_seq_len - Br * bid_z);

    const int q_block_local_start = Br * bid_z;
    const int q_block_local_end = ::min(target_seq_len - 1, q_block_local_start + Br - 1);
    const int q_block_abs_end = query_start_pos + q_block_local_end;

    //preparation-1: load Qi from GM to SM, and reset Oi to 0
    //Q[bid_y][Br * bid_z + tid_y][bid_x][*]
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
        O_sm_AT(tid_y, idx) = 0.0;
        Q_sm_AT(tid_y, idx) = 0.0;
        if(tid_y < bound_tid_y){
            Q_sm_AT(tid_y, idx) = to_float<T>(Q[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
        }
    }
    __syncthreads();

    //preparation-2: reset m_prev to -INFINITY and l_prev to 0
    if(tid_x == 0){
        m_prev[tid_y] = -8192.0;
        l_prev[tid_y] = 0.0;
    }
    __syncthreads();
    /****************************end-of-preparation*************************/


    /****************************main-loop**************************/
    #pragma unroll 4
    for(int j = 0; j < Tc; ++j){//对于每个K/V分块
        const int k_block_start = Bc * j;
        if (is_causal && k_block_start > q_block_abs_end) {
            // 当前 K block 已经完全位于所有 Q 后面; 因为后续 j 更大，所以后续 K block也一定全部不可见
            // 因此可以直接 break，而不是 continue; 
            break;
        }

        SP_AT(tid_y, tid_x) = -8192.0;
        __syncthreads();
        int bound_tid_x = ::min(Bc, src_seq_len - Bc * j);
    
        /*bool is_compute = true;//optimization: 分支处理,加速branch-resolving
        if (is_causal) {
            if (bid_z < j) {
                is_compute = false;  // 早期退出情况
            } else if (bid_z == j) {
                is_compute = (tid_y >= tid_x);  // 对角线以上
            }
        }*/

        const int q_pos = query_start_pos + Br * bid_z + tid_y;
        const int k_pos = Bc * j + tid_x;
        bool is_compute = !is_causal || (k_pos <= q_pos);

        //step-1: load Ki, Vi from GM to SM, reset Oi to 0
        //K[bid_y][Bc * j + tid_y][bid_x / p][*], V[bid_y][Bc * j + tid_y][bid_x / p][*]
        #pragma unroll
        for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
            K_T_sm_AT(idx, tid_y) = 0.0;
            V_sm_AT(tid_y, idx) = 0.0;
            if(tid_y < bound_tid_x){//注意这里是bound_tid_x
                K_T_sm_AT(idx, tid_y) = to_float<T>(K[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
                V_sm_AT(tid_y, idx) = to_float<T>(V[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
            }
        }
        __syncthreads();

        //step-2: S = Q @ K.T, point-wise
        if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
            float val0 = 0.0;//临时sum
            if(is_compute){
                #pragma unroll
                for(int k = 0; k < head_dim; ++k){
                    val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x);
                }
                SP_AT(tid_y, tid_x) = double(val0) * scale_factor;//必须用double,对精度影响最大的计算步骤
            }
        }
        __syncthreads();

        //step-3: m_new = max(m_prev, rowMax(S))
        float val1 = float(SP_AT(tid_y, tid_x));
        val1 = warp_reduce_max(val1);
        if(tid_x == 0 && tid_y < bound_tid_y){
            /*double val1 = SP_AT(tid_y, 0);//手动实现非并行求行最大值
            for(int h = 1; h < Bc; ++h){
                val1 = (val1 < SP_AT(tid_y, h)) ? SP_AT(tid_y, h) : val1;
            }*/
            m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
        }
        __syncthreads();

        //step-4: P = exp(S - m_new), point-wise
        if(tid_y < bound_tid_y && tid_x < bound_tid_x){
            if(is_compute){
                SP_AT(tid_y, tid_x) = myexp<double>(SP_AT(tid_y, tid_x) - double(m_new[tid_y]));
            }
            else{
                SP_AT(tid_y, tid_x) = 0.0;
            }
        }
        else{
            SP_AT(tid_y, tid_x) = 0.0;
        }
        __syncthreads();

        //step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
        float val2 = float(SP_AT(tid_y, tid_x));
        val2 = warp_reduce_sum(val2);
        float exp_result = myexp<float>(m_prev[tid_y] - m_new[tid_y]);
        if(tid_x == 0 && tid_y < bound_tid_y){
            /*double val2 = 0.0;//手动实现非并行求rowSum
            for(int h = 0; h < Bc; ++h){
                val2 += SP_AT(tid_y, h);
            }*/
            l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
        }
        __syncthreads();

        //step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
        //if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
        // 这里原来有bug: 应该是tid_y部分参与(对应部分行, 即尾部落单的token), 但是tid_x全都需要参与, 所有tid_x均摊处理head_dim
        // 如果tid_x也是部分参与, 那么就会有周期性的head_dim未参与最终的O更新; 比如只有head_dim 0~3, 32~35, ... 进行了更新
        if(tid_y < bound_tid_y){//32路并行计算Oi的每一行; 
            for(int u = tid_x; u < head_dim; u += blockDim.x){
                float val3 = 0.0;
                #pragma unroll
                for(int w = 0; w < Bc; ++w){//val3 += P[tid_y][w] * V[bid_y][Bc * j + w][bid_x / p][u];
                    val3 += float(SP_AT(tid_y, w)) * V_sm_AT(w, u);
                }
                O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
            }
        }
        __syncthreads();
      
        //step-7: m_prev <- m_new; l_prev <- l_new
        if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
            m_prev[tid_y] = m_new[tid_y];
            l_prev[tid_y] = l_new[tid_y];
        }
        __syncthreads();

    }
    /****************************end-of-main-loop**************************/

    /*****************************post-process****************************/
    //O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
    //O[bid_y][Br * bid_z + tid_y][bid_x][*]
    #pragma unroll
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
        if(tid_y < bound_tid_y){
            O[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx] = from_float<T>(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
        }
    }
    __syncthreads();
    /*****************************end-of-post-process****************************/

    //取消访问宏定义
    #undef   SP_AT
    #undef   Q_sm_AT
    #undef   K_T_sm_AT
    #undef   V_sm_AT
    #undef   O_sm_AT
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
    const dim3 grid_dim(q_heads, 1, grid_z); // #batch默认为1

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


// 在python环境下, 由于dtype是运行时决定的而非编译器决定, 因此这里无法使用模板参数, 而只能动态传参dtype_code
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






