#include <iostream>
#include <algorithm>
#include <random>
#include <cublas_v2.h>
#include <mma.h>


#define LOAD 0
#define STORE 1


//辅助函数: 协同加载/存储,即利用TROW x TCOL个线程将ROW x COL的数据块从GM加载至SM中/从SM存储至GM中, 其中TROW = blockDim.y, TCOL = blockDim.x
// 使用 __forceinline__ 强迫编译器内联，消除函数调用开销; 该函数可以直接使用blockDim等builtin参数
template <typename T, int TROW, int TCOL, int ROW, int COL, bool direction>//将ROW/COL作为模板常量而非参数可以提高相关变量运算的效率
__device__ __forceinline__ void cooperative_ldst(
    T* dst,                 // Shared Memory / Global Memoory 目的指针
    const T* src,           // Global Memory / Shared Memory 源指针
    int start_row,          // 当前需要加载的 Global 块的起始行
    int start_col,          // 当前需要加载的 Global 块的起始列
    int max_rows,           // 矩阵的真实行数边界
    int max_cols            // 矩阵的真实列数边界
) {
    // 1. 获取当前线程在 Block 内的一维线性 ID
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 2. 计算线程总数和需要搬运的总元素数
    const int num_threads = TROW * TCOL;
    const int num_elements = ROW * COL;

    // 3. 循环跳跃搬运 (如果线程数 >= 元素数，循环只会执行一次)
    #pragma unroll
    for (int i = flat_tid; i < num_elements; i += num_threads) {//每个线程循环搬运(跨步 = num_threads)直至全部搬运完毕
        int smem_row = i / COL;//映射为sharedmem访问坐标
        int smem_col = i % COL;

        int gm_row = start_row + smem_row;//再映射为globalmem访问坐标
        int gm_col = start_col + smem_col;

        if(direction){//store
            
        }
        if (gm_row < max_rows && gm_col < max_cols) {
            if(direction){//store
                dst[gm_row * max_cols + gm_col] = src[i];
            }
            else{
                dst[i] = src[gm_row * max_cols + gm_col];
            }
        } else {
            if(!direction){
                dst[i] = T(0); // 越界补零 (Padding)
            }
        }
    }
}


//辅助函数: 带列方向padding的协同加载/存储; shared memory物理列数为COL + PAD, 逻辑搬运范围仍为ROW x COL
template <typename T, int TROW, int TCOL, int ROW, int COL, int PAD, bool direction>
__device__ __forceinline__ void cooperative_ldst(
    T* dst,
    const T* src,
    int start_row,
    int start_col,
    int max_rows,
    int max_cols
) {
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int num_threads = TROW * TCOL;
    const int num_elements = ROW * COL;
    const int SMEM_COL = COL + PAD;

    #pragma unroll
    for (int i = flat_tid; i < num_elements; i += num_threads) {
        int smem_row = i / COL;
        int smem_col = i % COL;
        int smem_offset = smem_row * SMEM_COL + smem_col;

        int gm_row = start_row + smem_row;
        int gm_col = start_col + smem_col;

        if (gm_row < max_rows && gm_col < max_cols) {
            if(direction){
                dst[gm_row * max_cols + gm_col] = src[smem_offset];
            }
            else{
                dst[smem_offset] = src[gm_row * max_cols + gm_col];
            }
        } else {
            if(!direction){
                dst[smem_offset] = T(0);
            }
        }
    }
}




//辅助函数: float4向量化协同加载; 逻辑搬运范围为ROW x COL个float, 且COL必须为4的倍数
template <int TROW, int TCOL, int ROW, int COL, bool direction>
__device__ __forceinline__ void cooperative_ldst_vector(
    float* dst,
    const float* src,
    int start_row,
    int start_col,
    int max_rows,
    int max_cols
) {
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int num_threads = TROW * TCOL;
    const int vec_cols = COL / 4;
    const int num_vec_elements = ROW * vec_cols;

    for(int i = flat_tid; i < num_vec_elements; i += num_threads){
        int smem_row = i / vec_cols;
        int smem_vec_col = i % vec_cols;
        int smem_col = smem_vec_col * 4;
        int smem_offset = smem_row * COL + smem_col;

        int gm_row = start_row + smem_row;
        int gm_col = start_col + smem_col;

        if(direction){

        }
        else{
            if(gm_row < max_rows && gm_col + 3 < max_cols && gm_col % 4 == 0){
                float4 value = *reinterpret_cast<const float4*>(&src[gm_row * max_cols + gm_col]);
                *reinterpret_cast<float4*>(&dst[smem_offset]) = value;
            }
            else{
                #pragma unroll
                for(int j = 0; j < 4; ++j){
                    int col = gm_col + j;
                    if(gm_row < max_rows && col < max_cols){
                        dst[smem_offset + j] = src[gm_row * max_cols + col];
                    }
                    else{
                        dst[smem_offset + j] = 0.0f;
                    }
                }
            }
        }
    }
}


template <int BYTES>
__device__ __forceinline__ void cp_async_ca_shared_global(
    float* smem_ptr,
    const float* gmem_ptr,
    int src_bytes
) {
    unsigned smem_addr;
    asm volatile(
        "{ .reg .u64 smem_addr_u64;\n"
        "  cvta.to.shared.u64 smem_addr_u64, %1;\n"
        "  cvt.u32.u64 %0, smem_addr_u64;\n"
        "}\n"
        : "=r"(smem_addr)
        : "l"(smem_ptr)
    );

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "n"(BYTES), "r"(src_bytes)
    );
}


//辅助函数: float4向量化异步协同加载; 只发起cp.async, 需要调用方commit/wait后再读取shared memory
template <int TROW, int TCOL, int ROW, int COL, bool direction>
__device__ __forceinline__ void cooperative_ldst_vector_async(
    float* dst,
    const float* src,
    int start_row,
    int start_col,
    int max_rows,
    int max_cols
) {
    int flat_tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int num_threads = TROW * TCOL;
    const int vec_cols = COL / 4;
    const int num_vec_elements = ROW * vec_cols;

    for(int i = flat_tid; i < num_vec_elements; i += num_threads){
        int smem_row = i / vec_cols;
        int smem_vec_col = i % vec_cols;
        int smem_col = smem_vec_col * 4;
        int smem_offset = smem_row * COL + smem_col;

        int gm_row = start_row + smem_row;
        int gm_col = start_col + smem_col;

        if(direction){

        }
        else{
            if(gm_row < max_rows && gm_col + 3 < max_cols && gm_col % 4 == 0){//边界对齐
                cp_async_ca_shared_global<16>(&dst[smem_offset], &src[gm_row * max_cols + gm_col], 16);
            }
            else{
                #pragma unroll
                for(int j = 0; j < 4; ++j){//处理不对齐边界
                    int col = gm_col + j;
                    bool valid = gm_row < max_rows && col < max_cols;
                    const float* gmem_ptr = valid ? &src[gm_row * max_cols + col] : src;
                    cp_async_ca_shared_global<4>(&dst[smem_offset + j], gmem_ptr, valid ? 4 : 0);
                }
            }
        }
    }
}




//naive, 仅确保无bank conflict, 但没有使用shared memory
template <typename T>
__global__ void matmul1(T* C, const T* A, const T* B, const int M, const int K, const int N){//K为共同维度, 即(M, K) @ (K, N)
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //thread (idx, idy)负责处理计算C[idy][idx]的结果
    T sum = T(0);
    
    //C[idy][idx] = sum;
    if(idy < M && idx < N){//避免边缘的线程参与无效工作
        for(int k = 0; k < K; ++k){
            //sum += A[idy][k] * B[k][idx];
            sum += A[idy * K + k] * B[k * N + idx];
        }
        C[idy * N + idx] = sum;
    }
}





//添加shared memory tiling, 解决GM -> SM带宽瓶颈
template <typename T>
__global__ void matmul2(T* C, const T* A, const T* B, const int M, const int K, const int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //设置32x32的shared memory tile
    const int TILE_M = 32;//即blockDim.y
    const int TILE_N = 32;//即blockDim.x
    const int TILE_K = 32;//K方向上分块
    __shared__ T A_tile [TILE_M][TILE_K];
    __shared__ T B_tile [TILE_K][TILE_N];
    T sum = T(0);


    for(int i = 0; i < K; i += TILE_K){
        //cooperative loading A_tile/B_tile from GM to SM
        cooperative_ldst<T, TILE_M, TILE_N, TILE_M, TILE_K, 0>((T*)A_tile, A, blockIdx.y * blockDim.y, i, M, K);
        cooperative_ldst<T, TILE_M, TILE_N, TILE_K, TILE_N, 0>((T*)B_tile, B, i, blockIdx.x * blockDim.x, K, N);
        __syncthreads();//必须同步

        //进行A_tile @ B_tile矩阵乘, 只得到一个数
        for(int j = 0; j < TILE_K; ++j){
            sum += A_tile[tidy][j] * B_tile[j][tidx];
        }
        __syncthreads();
    }

    if(idy < M && idx < N){
        C[idy * N + idx] = sum;
    }
}







//再添加thread tiling, 解决SM -> 寄存器带宽瓶颈;但由此会带来非合并访存问题,因为一个线程处理4x4的tile, 因而写回的时候会存在跨步访存
template <typename T>
__global__ void matmul3(T* C, const T* A, const T* B, const int M, const int K, const int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int TILE_M = 8;//blockDim.y
    const int TILE_N = 32;//blockDim.x
    const int TILE_THREAD_M = 4;
    const int TILE_THREAD_N = 4;//每个thread要负责4x4窗口计算; 注意,gridDim.x = N/(TILE_N * TILE_THREAD_N), gridDim.y = M/(TILE_M * TILE_THREAD_M)
    const int TILE_K = 16;//TILE_K不能太大，否则smem会溢出

    __shared__ T A_tile [TILE_M * TILE_THREAD_M][TILE_K];
    __shared__ T B_tile [TILE_K][TILE_N * TILE_THREAD_N];

    //这些都是一个thread所私有的,因而无法通过协同存储的方式存回GM
    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    for(int i = 0; i < K; i += TILE_K){
        //cooperative loading A_tile & B_tile
        //cooperative_ldst<T, TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((T*)A_tile, A, blockIdx.y * (TILE_M * TILE_THREAD_M), i, M, K);
        //cooperative_ldst<T, TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((T*)B_tile, B, i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
        cooperative_ldst_vector<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile, A, blockIdx.y * (TILE_M * TILE_THREAD_M), i, M, K);
        cooperative_ldst_vector<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile, B, i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
        __syncthreads();

        for(int k = 0; k < TILE_K; ++k){//原本只要进行一个数的结果计算,现在需要负责4x4的结果计算
            for(int thread = 0; thread < TILE_THREAD_M; ++thread){
                A_reg[thread] = A_tile[tidy * TILE_THREAD_M + thread][k];
            }
            for(int thread = 0; thread < TILE_THREAD_N; ++thread){
                B_reg[thread] = B_tile[k][tidx * TILE_THREAD_N + thread];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        __syncthreads();
    }

    //write back
    /*for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; ++n){
            int gm_row = blockIdx.y * (TILE_M * TILE_THREAD_M) + tidy * TILE_THREAD_M + m;
            int gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + n;
            if(gm_row < M && gm_col < N){
                C[gm_row * N + gm_col] = thread_accum[m][n];
            }
        }
    }*/

    //vectorized write back
    for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; n += 4){//强制向量化写回
            int gm_row = blockIdx.y * (TILE_M * TILE_THREAD_M) + tidy * TILE_THREAD_M + m;
            int gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + n;
            int gm_offset = gm_row * N + gm_col;

            if(gm_row < M && gm_col + 3 < N && gm_offset % 4 == 0){
                float4 values = make_float4(
                    thread_accum[m][n],
                    thread_accum[m][n + 1],
                    thread_accum[m][n + 2],
                    thread_accum[m][n + 3]
                );
                reinterpret_cast<float4*>(C)[gm_offset / 4] = values;
            }
            else{
                for(int nn = n; nn < n + 4 && nn < TILE_THREAD_N; ++nn){
                    int scalar_gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + nn;
                    if(gm_row < M && scalar_gm_col < N){
                        C[gm_row * N + scalar_gm_col] = thread_accum[m][nn];
                    }
                }
            }
        }
    }

}


//在matmul3基础上使用cp.async和double buffering隐藏GM -> SM搬运延迟
template <typename T>
__global__ void matmul3_db(T* C, const T* A, const T* B, const int M, const int K, const int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int TILE_M = 8;//blockDim.y
    const int TILE_N = 32;//blockDim.x
    const int TILE_THREAD_M = 4;
    const int TILE_THREAD_N = 4;//每个thread要负责4x4窗口计算; 注意,gridDim.x = N/(TILE_N * TILE_THREAD_N), gridDim.y = M/(TILE_M * TILE_THREAD_M)
    const int TILE_K = 16;//TILE_K不能太大，否则smem会溢出

    __shared__ T A_tile [2][TILE_M * TILE_THREAD_M][TILE_K];
    __shared__ T B_tile [2][TILE_K][TILE_N * TILE_THREAD_N];

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    //先等待第一块数据加载完毕
    int curr_stage = 0;
    cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[curr_stage], A, blockIdx.y * (TILE_M * TILE_THREAD_M), 0, M, K);
    cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[curr_stage], B, 0, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
    asm volatile("cp.async.commit_group;\n");
    //asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    for(int i = 0; i < K; i += TILE_K){
        curr_stage = (i / TILE_K) & 1;
        int next_stage = curr_stage ^ 1;
        int next_i = i + TILE_K;

        //循环内,先加载下一块
        if(next_i < K){
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[next_stage], A, blockIdx.y * (TILE_M * TILE_THREAD_M), next_i, M, K);
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[next_stage], B, next_i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
            asm volatile("cp.async.commit_group;\n");
        }

        //循环内,紧接着计算本块
        for(int k = 0; k < TILE_K; ++k){
            for(int thread = 0; thread < TILE_THREAD_M; ++thread){
                A_reg[thread] = A_tile[curr_stage][tidy * TILE_THREAD_M + thread][k];
            }
            for(int thread = 0; thread < TILE_THREAD_N; ++thread){
                B_reg[thread] = B_tile[curr_stage][k][tidx * TILE_THREAD_N + thread];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        if(next_i < K){
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
        }
    }

    //vectorized write back
    for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; n += 4){
            int gm_row = blockIdx.y * (TILE_M * TILE_THREAD_M) + tidy * TILE_THREAD_M + m;
            int gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + n;
            int gm_offset = gm_row * N + gm_col;

            if(gm_row < M && gm_col + 3 < N && gm_offset % 4 == 0){
                float4 values = make_float4(
                    thread_accum[m][n],
                    thread_accum[m][n + 1],
                    thread_accum[m][n + 2],
                    thread_accum[m][n + 3]
                );
                reinterpret_cast<float4*>(C)[gm_offset / 4] = values;
            }
            else{
                for(int nn = n; nn < n + 4 && nn < TILE_THREAD_N; ++nn){
                    int scalar_gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + nn;
                    if(gm_row < M && scalar_gm_col < N){
                        C[gm_row * N + scalar_gm_col] = thread_accum[m][nn];
                    }
                }
            }
        }
    }
}






//在matmul3基础上使用cp.async和double buffering隐藏GM -> SM搬运延迟
template <typename T>
__global__ void matmul3_mb(T* C, const T* A, const T* B, const int M, const int K, const int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int TILE_M = 8;//blockDim.y
    const int TILE_N = 32;//blockDim.x
    const int TILE_THREAD_M = 4;
    const int TILE_THREAD_N = 4;//每个thread要负责4x4窗口计算; 注意,gridDim.x = N/(TILE_N * TILE_THREAD_N), gridDim.y = M/(TILE_M * TILE_THREAD_M)
    const int TILE_K = 16;//TILE_K不能太大，否则smem会溢出

    __shared__ T A_tile [2][TILE_M * TILE_THREAD_M][TILE_K];
    __shared__ T B_tile [2][TILE_K][TILE_N * TILE_THREAD_N];

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    //prologue: 预加载第一块,或者N-1块对于N级流水
    int curr_stage = 0;
    cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[curr_stage], A, blockIdx.y * (TILE_M * TILE_THREAD_M), 0, M, K);
    cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[curr_stage], B, 0, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
    asm volatile("cp.async.commit_group;\n");
    //asm volatile("cp.async.wait_group 0;\n");
    //__syncthreads();


    //main-loop
    for(int i = 0; i < K; i += TILE_K){
        curr_stage = (i / TILE_K) & 1;
        int next_stage = curr_stage ^ 1;
        int next_i = i + TILE_K;

        //循环内,先加载下一块
        if(next_i < K){
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[next_stage], A, blockIdx.y * (TILE_M * TILE_THREAD_M), next_i, M, K);
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[next_stage], B, next_i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        }
        else{//最后一块,没有下一块
            asm volatile("cp.async.wait_group 0;\n");//必须等待之前所有group都加载完毕
        }

        __syncthreads();

        //循环内,紧接着计算本块
        for(int k = 0; k < TILE_K; ++k){
            for(int thread = 0; thread < TILE_THREAD_M; ++thread){
                A_reg[thread] = A_tile[curr_stage][tidy * TILE_THREAD_M + thread][k];
            }
            for(int thread = 0; thread < TILE_THREAD_N; ++thread){
                B_reg[thread] = B_tile[curr_stage][k][tidx * TILE_THREAD_N + thread];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        __syncthreads();
        
    }


    //epilogue: vectorized write back
    for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; n += 4){
            int gm_row = blockIdx.y * (TILE_M * TILE_THREAD_M) + tidy * TILE_THREAD_M + m;
            int gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + n;
            int gm_offset = gm_row * N + gm_col;

            if(gm_row < M && gm_col + 3 < N && gm_offset % 4 == 0){
                float4 values = make_float4(
                    thread_accum[m][n],
                    thread_accum[m][n + 1],
                    thread_accum[m][n + 2],
                    thread_accum[m][n + 3]
                );
                reinterpret_cast<float4*>(C)[gm_offset / 4] = values;
            }
            else{
                for(int nn = n; nn < n + 4 && nn < TILE_THREAD_N; ++nn){
                    int scalar_gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + nn;
                    if(gm_row < M && scalar_gm_col < N){
                        C[gm_row * N + scalar_gm_col] = thread_accum[m][nn];
                    }
                }
            }
        }
    }
}








//在采用4-stage流水线进行matmul
template <typename T>
__global__ void matmul3_4b(T* C, const T* A, const T* B, const int M, const int K, const int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int TILE_M = 8;//blockDim.y
    const int TILE_N = 32;//blockDim.x
    const int TILE_THREAD_M = 4;
    const int TILE_THREAD_N = 4;//每个thread要负责4x4窗口计算; 注意,gridDim.x = N/(TILE_N * TILE_THREAD_N), gridDim.y = M/(TILE_M * TILE_THREAD_M)
    const int TILE_K = 8;//TILE_K不能太大，否则smem会溢出
    const int STAGE = 4;
    const int TILE_NUM = (K + TILE_K - 1)/TILE_K;
    int drain_count = STAGE - 1;

    if(TILE_NUM < STAGE - 1){
        return;//K_tile数量小于3, prologue会访存越界, 直接返回
    }

    __shared__ T A_tile [STAGE][TILE_M * TILE_THREAD_M][TILE_K];
    __shared__ T B_tile [STAGE][TILE_K][TILE_N * TILE_THREAD_N];

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    //prologue: 预加载第一块,或者前N-1块对于N级流水
    #pragma unroll
    for(int i = 0; i < STAGE - 1; ++i){
        cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[i], A, blockIdx.y * (TILE_M * TILE_THREAD_M), TILE_K * i, M, K);
        cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[i], B, TILE_K * i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
        asm volatile("cp.async.commit_group;\n");
    }

    //main-loop
    for(int i = 0; i < K; i += TILE_K){
        int curr_stage = (i / TILE_K) % STAGE;
        int next_stage = (curr_stage + STAGE - 1) % STAGE;
        int next_i = i + TILE_K * (STAGE - 1);
        
        //steady mainloop
        if(next_i < K){
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_M * TILE_THREAD_M, TILE_K, 0>((float*)A_tile[next_stage], A, blockIdx.y * (TILE_M * TILE_THREAD_M), next_i, M, K);
            cooperative_ldst_vector_async<TILE_M, TILE_N, TILE_K, TILE_N * TILE_THREAD_N, 0>((float*)B_tile[next_stage], B, next_i, blockIdx.x * (TILE_N * TILE_THREAD_N), K, N);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 3;\n");
        }
        else{//tail drain
            drain_count --;
            switch(drain_count){
                case 2: asm volatile("cp.async.wait_group 2;\n"); break;
                case 1: asm volatile("cp.async.wait_group 1;\n"); break;
                case 0: asm volatile("cp.async.wait_group 0;\n"); break;
            }
        }    
            
        __syncthreads();
        for(int k = 0; k < TILE_K; ++k){
            for(int thread = 0; thread < TILE_THREAD_M; ++thread){
                A_reg[thread] = A_tile[curr_stage][tidy * TILE_THREAD_M + thread][k];
            }
            for(int thread = 0; thread < TILE_THREAD_N; ++thread){
                B_reg[thread] = B_tile[curr_stage][k][tidx * TILE_THREAD_N + thread];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }
        __syncthreads();
        
    }


    //epilogue: vectorized write back
    for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; n += 4){
            int gm_row = blockIdx.y * (TILE_M * TILE_THREAD_M) + tidy * TILE_THREAD_M + m;
            int gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + n;
            int gm_offset = gm_row * N + gm_col;

            if(gm_row < M && gm_col + 3 < N && gm_offset % 4 == 0){
                float4 values = make_float4(
                    thread_accum[m][n],
                    thread_accum[m][n + 1],
                    thread_accum[m][n + 2],
                    thread_accum[m][n + 3]
                );
                reinterpret_cast<float4*>(C)[gm_offset / 4] = values;
            }
            else{
                for(int nn = n; nn < n + 4 && nn < TILE_THREAD_N; ++nn){
                    int scalar_gm_col = blockIdx.x * (TILE_N * TILE_THREAD_N) + tidx * TILE_THREAD_N + nn;
                    if(gm_row < M && scalar_gm_col < N){
                        C[gm_row * N + scalar_gm_col] = thread_accum[m][nn];
                    }
                }
            }
        }
    }
}



















//在thread tiling的基础上继续解决写回时的跨步访存问题
template <typename T>
__global__ void matmul4(T* C, const T* A, const T* B, const int M, const int K, const int N){
    const int BLOCK_THREAD_M = 8;//blockDim.y
    const int BLOCK_THREAD_N = 32;//blockDim.x

    const int TILE_M = 32;//block tile M
    const int TILE_N = 128;//block tile N
    const int TILE_K = 8;//K方向上分块

    const int TILE_WARP_M = 16;//warp tile M
    const int TILE_WARP_N = 32;//warp tile N

    const int TILE_THREAD_M = 16;//thread tile M
    const int TILE_THREAD_N = 1;//thread tile N

    const int WARPS_M = TILE_M / TILE_WARP_M;
    const int WARPS_N = TILE_N / TILE_WARP_N;
    const int WARP_THREADS_M = TILE_WARP_M / TILE_THREAD_M;
    const int WARP_THREADS_N = TILE_WARP_N / TILE_THREAD_N;

    static_assert(TILE_M % TILE_WARP_M == 0, "TILE_M must be divisible by TILE_WARP_M");
    static_assert(TILE_N % TILE_WARP_N == 0, "TILE_N must be divisible by TILE_WARP_N");
    static_assert(TILE_WARP_M % TILE_THREAD_M == 0, "TILE_WARP_M must be divisible by TILE_THREAD_M");
    static_assert(TILE_WARP_N % TILE_THREAD_N == 0, "TILE_WARP_N must be divisible by TILE_THREAD_N");
    static_assert(WARP_THREADS_M * WARP_THREADS_N == 32, "one warp tile must be covered by 32 threads");
    static_assert(WARPS_M * WARPS_N * 32 == BLOCK_THREAD_M * BLOCK_THREAD_N, "block tile must match block thread count");

    __shared__ T A_tile [TILE_M][TILE_K];
    __shared__ T B_tile [TILE_K][TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int lane_m = lane_id / WARP_THREADS_N;
    int lane_n = lane_id % WARP_THREADS_N;

    int thread_tile_row = warp_m * TILE_WARP_M + lane_m * TILE_THREAD_M;
    int thread_tile_col = warp_n * TILE_WARP_N + lane_n * TILE_THREAD_N;

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    for(int i = 0; i < K; i += TILE_K){
        //GM -> SM仍然由整个block协同加载, warp tiling主要影响SM -> register和计算映射
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_M, TILE_K, 0>((T*)A_tile, A, blockIdx.y * TILE_M, i, M, K);
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_K, TILE_N, 0>((T*)B_tile, B, i, blockIdx.x * TILE_N, K, N);
        __syncthreads();

        for(int k = 0; k < TILE_K; ++k){
            for(int m = 0; m < TILE_THREAD_M; ++m){
                A_reg[m] = A_tile[thread_tile_row + m][k];
            }
            for(int n = 0; n < TILE_THREAD_N; ++n){
                B_reg[n] = B_tile[k][thread_tile_col + n];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        __syncthreads();
    }

    //每个warp负责一个TILE_WARP_M x TILE_WARP_N的连续C子块, lane按列连续写回
    for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; ++n){
            int gm_row = blockIdx.y * TILE_M + thread_tile_row + m;
            int gm_col = blockIdx.x * TILE_N + thread_tile_col + n;
            if(gm_row < M && gm_col < N){
                C[gm_row * N + gm_col] = thread_accum[m][n];
            }
        }
    }
}


//按照Block Tile -> Warp Tile -> Thread Tile层级划分, 每个thread负责一个连续8x8的C子块
template <typename T>
__global__ void matmul5(T* C, const T* A, const T* B, const int M, const int K, const int N){
    const int BLOCK_THREAD_M = 8;//blockDim.y
    const int BLOCK_THREAD_N = 32;//blockDim.x

    const int TILE_M = 64;//block tile M
    const int TILE_N = 64;//block tile N
    const int TILE_K = 8;//K方向上分块

    const int TILE_WARP_M = 32;//warp tile M
    const int TILE_WARP_N = 16;//warp tile N

    const int TILE_THREAD_M = 4;//thread tile M
    const int TILE_THREAD_N = 4;//thread tile N

    const int WARPS_M = TILE_M / TILE_WARP_M;//2
    const int WARPS_N = TILE_N / TILE_WARP_N;//2
    const int WARP_THREADS_M = TILE_WARP_M / TILE_THREAD_M;//8
    const int WARP_THREADS_N = TILE_WARP_N / TILE_THREAD_N;//4

    __shared__ T A_tile [TILE_M][TILE_K];
    __shared__ T B_tile [TILE_K][TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int lane_m = lane_id / WARP_THREADS_N;
    int lane_n = lane_id % WARP_THREADS_N;

    int thread_tile_row = warp_m * TILE_WARP_M + lane_m * TILE_THREAD_M;
    int thread_tile_col = warp_n * TILE_WARP_N + lane_n * TILE_THREAD_N;

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    for(int i = 0; i < K; i += TILE_K){
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_M, TILE_K, 0>((T*)A_tile, A, blockIdx.y * TILE_M, i, M, K);
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_K, TILE_N, 0>((T*)B_tile, B, i, blockIdx.x * TILE_N, K, N);
        __syncthreads();

        for(int k = 0; k < TILE_K; ++k){
            for(int m = 0; m < TILE_THREAD_M; ++m){
                A_reg[m] = A_tile[thread_tile_row + m][k];
            }
            for(int n = 0; n < TILE_THREAD_N; ++n){
                B_reg[n] = B_tile[k][thread_tile_col + n];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        __syncthreads();
    }

    /*for(int m = 0; m < TILE_THREAD_M; ++m){
        for(int n = 0; n < TILE_THREAD_N; ++n){
            int gm_row = blockIdx.y * TILE_M + thread_tile_row + m;
            int gm_col = blockIdx.x * TILE_N + thread_tile_col + n;
            if(gm_row < M && gm_col < N){
                C[gm_row * N + gm_col] = thread_accum[m][n];
            }
        }
    }*/
    //写回直接特化为4次float4;由于原矩阵尺寸为4的倍数因而可以直接这么写,但是如果存在边界情况则需要分类
    int gm_row0 = blockIdx.y * TILE_M + thread_tile_row + 0;
    int gm_row1 = blockIdx.y * TILE_M + thread_tile_row + 1;
    int gm_row2 = blockIdx.y * TILE_M + thread_tile_row + 2;
    int gm_row3 = blockIdx.y * TILE_M + thread_tile_row + 3;
    int gm_col = blockIdx.x * TILE_N + thread_tile_col + 0;
    float4 row0 = make_float4(float(thread_accum[0][0]), float(thread_accum[0][1]), float(thread_accum[0][2]), float(thread_accum[0][3]));
    float4 row1 = make_float4(float(thread_accum[1][0]), float(thread_accum[1][1]), float(thread_accum[1][2]), float(thread_accum[1][3]));
    float4 row2 = make_float4(float(thread_accum[2][0]), float(thread_accum[2][1]), float(thread_accum[2][2]), float(thread_accum[2][3]));
    float4 row3 = make_float4(float(thread_accum[3][0]), float(thread_accum[3][1]), float(thread_accum[3][2]), float(thread_accum[3][3]));
    *reinterpret_cast<float4*>(&C[gm_row0 * N + gm_col]) = row0;
    *reinterpret_cast<float4*>(&C[gm_row1 * N + gm_col]) = row1;
    *reinterpret_cast<float4*>(&C[gm_row2 * N + gm_col]) = row2;
    *reinterpret_cast<float4*>(&C[gm_row3 * N + gm_col]) = row3;

}


//在matmul5基础上给A_tile的K维末尾添加padding, 使A_tile计算阶段读访问的bank被打散
template <typename T>
__global__ void matmul7(T* C, const T* A, const T* B, const int M, const int K, const int N){
    const int BLOCK_THREAD_M = 8;//blockDim.y
    const int BLOCK_THREAD_N = 32;//blockDim.x

    const int TILE_M = 64;//block tile M
    const int TILE_N = 64;//block tile N
    const int TILE_K = 8;//K方向上分块
    const int A_PAD = 1;//A_tile物理列数为TILE_K + A_PAD, 用于消除A_tile读阶段bank conflict

    const int TILE_WARP_M = 32;//warp tile M
    const int TILE_WARP_N = 16;//warp tile N

    const int TILE_THREAD_M = 4;//thread tile M
    const int TILE_THREAD_N = 4;//thread tile N

    const int WARPS_M = TILE_M / TILE_WARP_M;
    const int WARPS_N = TILE_N / TILE_WARP_N;
    const int WARP_THREADS_M = TILE_WARP_M / TILE_THREAD_M;
    const int WARP_THREADS_N = TILE_WARP_N / TILE_THREAD_N;

    __shared__ T A_tile [TILE_M][TILE_K + A_PAD];
    __shared__ T B_tile [TILE_K][TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int lane_m = lane_id / WARP_THREADS_N;
    int lane_n = lane_id % WARP_THREADS_N;

    int thread_tile_row = warp_m * TILE_WARP_M + lane_m * TILE_THREAD_M;
    int thread_tile_col = warp_n * TILE_WARP_N + lane_n * TILE_THREAD_N;

    T A_reg[TILE_THREAD_M];
    T B_reg[TILE_THREAD_N];
    T thread_accum[TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    for(int i = 0; i < K; i += TILE_K){
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_M, TILE_K, A_PAD, 0>((T*)A_tile, A, blockIdx.y * TILE_M, i, M, K);
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_K, TILE_N, 0>((T*)B_tile, B, i, blockIdx.x * TILE_N, K, N);
        __syncthreads();

        for(int k = 0; k < TILE_K; ++k){
            for(int m = 0; m < TILE_THREAD_M; ++m){
                A_reg[m] = A_tile[thread_tile_row + m][k];
            }
            for(int n = 0; n < TILE_THREAD_N; ++n){
                B_reg[n] = B_tile[k][thread_tile_col + n];
            }
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    thread_accum[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }

        __syncthreads();
    }

    int gm_row0 = blockIdx.y * TILE_M + thread_tile_row + 0;
    int gm_row1 = blockIdx.y * TILE_M + thread_tile_row + 1;
    int gm_row2 = blockIdx.y * TILE_M + thread_tile_row + 2;
    int gm_row3 = blockIdx.y * TILE_M + thread_tile_row + 3;
    int gm_col = blockIdx.x * TILE_N + thread_tile_col + 0;
    float4 row0 = make_float4(float(thread_accum[0][0]), float(thread_accum[0][1]), float(thread_accum[0][2]), float(thread_accum[0][3]));
    float4 row1 = make_float4(float(thread_accum[1][0]), float(thread_accum[1][1]), float(thread_accum[1][2]), float(thread_accum[1][3]));
    float4 row2 = make_float4(float(thread_accum[2][0]), float(thread_accum[2][1]), float(thread_accum[2][2]), float(thread_accum[2][3]));
    float4 row3 = make_float4(float(thread_accum[3][0]), float(thread_accum[3][1]), float(thread_accum[3][2]), float(thread_accum[3][3]));
    *reinterpret_cast<float4*>(&C[gm_row0 * N + gm_col]) = row0;
    *reinterpret_cast<float4*>(&C[gm_row1 * N + gm_col]) = row1;
    *reinterpret_cast<float4*>(&C[gm_row2 * N + gm_col]) = row2;
    *reinterpret_cast<float4*>(&C[gm_row3 * N + gm_col]) = row3;
}


//在matmul7的warp tile布局基础上, 使用TF32 WMMA/MMA完成每个warp的32xK @ Kx16计算
__global__ void matmul8(float* C, const float* A, const float* B, const int M, const int K, const int N){
    const int BLOCK_THREAD_M = 8;
    const int BLOCK_THREAD_N = 32;

    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 8;

    const int TILE_WARP_M = 32;
    const int TILE_WARP_N = 16;

    const int WARPS_N = TILE_N / TILE_WARP_N;

    __shared__ float A_tile[TILE_M][TILE_K];
    __shared__ float B_tile[TILE_K][TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int warp_tile_row = warp_m * TILE_WARP_M;
    int warp_tile_col = warp_n * TILE_WARP_N;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> A_frag0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> A_frag1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> B_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> C_frag0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> C_frag1;

    nvcuda::wmma::fill_fragment(C_frag0, 0.0f);
    nvcuda::wmma::fill_fragment(C_frag1, 0.0f);

    for(int i = 0; i < K; i += TILE_K){
        #pragma unroll
        for(int elem = tid; elem < TILE_M * TILE_K; elem += BLOCK_THREAD_M * BLOCK_THREAD_N){
            int smem_row = elem / TILE_K;
            int smem_col = elem % TILE_K;
            int gm_row = blockIdx.y * TILE_M + smem_row;
            int gm_col = i + smem_col;

            if(gm_row < M && gm_col < K){
                A_tile[smem_row][smem_col] = nvcuda::wmma::__float_to_tf32(A[gm_row * K + gm_col]);
            }
            else{
                A_tile[smem_row][smem_col] = 0.0f;
            }
        }

        #pragma unroll
        for(int elem = tid; elem < TILE_K * TILE_N; elem += BLOCK_THREAD_M * BLOCK_THREAD_N){
            int smem_row = elem / TILE_N;
            int smem_col = elem % TILE_N;
            int gm_row = i + smem_row;
            int gm_col = blockIdx.x * TILE_N + smem_col;

            if(gm_row < K && gm_col < N){
                B_tile[smem_row][smem_col] = nvcuda::wmma::__float_to_tf32(B[gm_row * N + gm_col]);
            }
            else{
                B_tile[smem_row][smem_col] = 0.0f;
            }
        }

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(A_frag0, &A_tile[warp_tile_row][0], TILE_K);
        nvcuda::wmma::load_matrix_sync(A_frag1, &A_tile[warp_tile_row + 16][0], TILE_K);
        nvcuda::wmma::load_matrix_sync(B_frag, &B_tile[0][warp_tile_col], TILE_N);

        nvcuda::wmma::mma_sync(C_frag0, A_frag0, B_frag, C_frag0);
        nvcuda::wmma::mma_sync(C_frag1, A_frag1, B_frag, C_frag1);

        __syncthreads();
    }

    int gm_row0 = blockIdx.y * TILE_M + warp_tile_row;
    int gm_row1 = gm_row0 + 16;
    int gm_col = blockIdx.x * TILE_N + warp_tile_col;

    if(gm_row0 + 15 < M && gm_col + 15 < N){
        nvcuda::wmma::store_matrix_sync(&C[gm_row0 * N + gm_col], C_frag0, N, nvcuda::wmma::mem_row_major);
    }
    if(gm_row1 + 15 < M && gm_col + 15 < N){
        nvcuda::wmma::store_matrix_sync(&C[gm_row1 * N + gm_col], C_frag1, N, nvcuda::wmma::mem_row_major);
    }
}


//按照图中的Thread Tile布局, 每个thread负责4个跳步后的4x4 C子块
template <typename T>
__global__ void matmul6(T* C, const T* A, const T* B, const int M, const int K, const int N){
    const int BLOCK_THREAD_M = 8;//blockDim.y
    const int BLOCK_THREAD_N = 32;//blockDim.x

    const int TILE_M = 128;//block tile M
    const int TILE_N = 128;//block tile N
    const int TILE_K = 8;//K方向上分块

    const int TILE_WARP_M = 64;//warp tile M
    const int TILE_WARP_N = 32;//warp tile N

    const int TILE_THREAD_M = 4;//每个小thread tile的M方向大小
    const int TILE_THREAD_N = 4;//每个小thread tile的N方向大小
    const int THREAD_TILES_M = 2;//上下两个4x4
    const int THREAD_TILES_N = 2;//左右两个4x4

    const int WARPS_M = TILE_M / TILE_WARP_M;
    const int WARPS_N = TILE_N / TILE_WARP_N;
    const int WARP_THREADS_M = TILE_WARP_M / (TILE_THREAD_M * THREAD_TILES_M);
    const int WARP_THREADS_N = TILE_WARP_N / (TILE_THREAD_N * THREAD_TILES_N);
    const int THREAD_TILE_ROW_STRIDE = WARP_THREADS_M * TILE_THREAD_M;
    const int THREAD_TILE_COL_STRIDE = WARP_THREADS_N * TILE_THREAD_N;

    /*static_assert(TILE_M % TILE_WARP_M == 0, "TILE_M must be divisible by TILE_WARP_M");
    static_assert(TILE_N % TILE_WARP_N == 0, "TILE_N must be divisible by TILE_WARP_N");
    static_assert(TILE_WARP_M % (TILE_THREAD_M * THREAD_TILES_M) == 0, "TILE_WARP_M must match thread tile layout");
    static_assert(TILE_WARP_N % (TILE_THREAD_N * THREAD_TILES_N) == 0, "TILE_WARP_N must match thread tile layout");
    static_assert(WARP_THREADS_M * WARP_THREADS_N == 32, "one warp tile must be covered by 32 threads");
    static_assert(WARPS_M * WARPS_N * 32 == BLOCK_THREAD_M * BLOCK_THREAD_N, "block tile must match block thread count");
    */

    __shared__ T A_tile [TILE_M][TILE_K];
    __shared__ T B_tile [TILE_K][TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int lane_m = lane_id / WARP_THREADS_N;
    int lane_n = lane_id % WARP_THREADS_N;

    int thread_tile_row = warp_m * TILE_WARP_M + lane_m * TILE_THREAD_M;
    int thread_tile_col = warp_n * TILE_WARP_N + lane_n * TILE_THREAD_N;

    T A_reg[THREAD_TILES_M][TILE_THREAD_M];
    T B_reg[THREAD_TILES_N][TILE_THREAD_N];
    T thread_accum[THREAD_TILES_M][THREAD_TILES_N][TILE_THREAD_M][TILE_THREAD_N] = {T(0)};

    for(int i = 0; i < K; i += TILE_K){
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_M, TILE_K, 0>((T*)A_tile, A, blockIdx.y * TILE_M, i, M, K);
        cooperative_ldst<T, BLOCK_THREAD_M, BLOCK_THREAD_N, TILE_K, TILE_N, 0>((T*)B_tile, B, i, blockIdx.x * TILE_N, K, N);
        __syncthreads();

        for(int k = 0; k < TILE_K; ++k){
            for(int tm = 0; tm < THREAD_TILES_M; ++tm){
                for(int m = 0; m < TILE_THREAD_M; ++m){
                    A_reg[tm][m] = A_tile[thread_tile_row + tm * THREAD_TILE_ROW_STRIDE + m][k];
                }
            }
            for(int tn = 0; tn < THREAD_TILES_N; ++tn){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    B_reg[tn][n] = B_tile[k][thread_tile_col + tn * THREAD_TILE_COL_STRIDE + n];
                }
            }
            for(int tm = 0; tm < THREAD_TILES_M; ++tm){
                for(int tn = 0; tn < THREAD_TILES_N; ++tn){
                    for(int m = 0; m < TILE_THREAD_M; ++m){
                        for(int n = 0; n < TILE_THREAD_N; ++n){
                            thread_accum[tm][tn][m][n] += A_reg[tm][m] * B_reg[tn][n];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    for(int tm = 0; tm < THREAD_TILES_M; ++tm){
        for(int tn = 0; tn < THREAD_TILES_N; ++tn){
            for(int m = 0; m < TILE_THREAD_M; ++m){
                for(int n = 0; n < TILE_THREAD_N; ++n){
                    int gm_row = blockIdx.y * TILE_M + thread_tile_row + tm * THREAD_TILE_ROW_STRIDE + m;
                    int gm_col = blockIdx.x * TILE_N + thread_tile_col + tn * THREAD_TILE_COL_STRIDE + n;
                    if(gm_row < M && gm_col < N){
                        C[gm_row * N + gm_col] = thread_accum[tm][tn][m][n];
                    }
                }
            }
        }
    }
}








// 利用cublas的cublasSgemm函数进行矩阵乘法: C = A * B
// S代表单精度，A的大小为 M x K，B的大小为 K x N，C的大小为 M x N
// 注意：C++中矩阵是按行主序存储，而cuBLAS期望的是列主序存储。
void matmul_cublas(float* C, const float* A, const float* B, const int M, const int N, const int K, cublasHandle_t handle) {
    // 设置标量参数
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 设定 cuBLAS 视角下的维度与 Leading Dimension (主维度)
    // 根据 C^T = B^T * A^T，我们将 C++ 的 B 作为第一个操作数，A 作为第二个操作数传给 cuBLAS
    
    // 在 cuBLAS 视角下（列主序），矩阵的维度等于它在 C++ 行主序下的转置维度
    int m_cublas = N; // C++ B的列数 -> cuBLAS视角下第一个矩阵的行数
    int n_cublas = M; // C++ A的行数 -> cuBLAS视角下第二个矩阵的列数
    int k_cublas = K; // 内侧相乘维度

    // 设定 Leading Dimension (主维度)
    // 在 C++ 的行优先中，主维度 lda/ldb/ldc 实际上就是矩阵的“物理列数（宽度）”
    int ldb = N; // C++ 矩阵 B 的物理列数
    int lda = K; // C++ 矩阵 A 的物理列数
    int ldc = N; // C++ 矩阵 C 的物理列数

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, // 对传入的第一个矩阵（其实是B）不做额外操作
        CUBLAS_OP_N, // 对传入的第二个矩阵（其实是A）不做额外操作
        m_cublas,    // 输出矩阵在 cuBLAS 视角下的行数 (N)
        n_cublas,    // 输出矩阵在 cuBLAS 视角下的列数 (M)
        k_cublas,    // 参与相乘的内维度 (K)
        &alpha,
        B, ldb,      // 注意：这里第一个操作数传 B
        A, lda,      // 注意：这里第二个操作数传 A
        &beta,
        C, ldc       // 输出到 C
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS sgemm failed!" << std::endl;
    }
}




int main(){
    int M = 4096, K = 4096, N = 4096;

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    size_t totalByte_A = M * K * sizeof(float);
    size_t totalByte_B = K * N * sizeof(float);
    size_t totalByte_C = M * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    //随机初始化
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // 使用 std::generate 结合 Lambda 表达式进行优雅赋值
    std::generate(h_A, h_A + M * K, [&]() { return dist(gen); });
    std::generate(h_B, h_B + K * N, [&]() { return dist(gen); });


    //step1: device侧分配资源
    cudaMalloc((void **)&d_A, totalByte_A);
    cudaMalloc((void **)&d_B, totalByte_B);
    cudaMalloc((void **)&d_C, totalByte_C);

    //step2: copy H2D
    cudaMemcpy(d_A, h_A, totalByte_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalByte_B, cudaMemcpyHostToDevice);

    //step3: launch kernel
    dim3 block_dim1(32, 32);
    dim3 grid_dim1((N + 31)/32, (M + 31)/32);
    dim3 block_dim2(32, 32);
    dim3 grid_dim2((N + 31)/32, (M + 31)/32);
    dim3 block_dim3(32, 8);
    dim3 grid_dim3((N + 127)/128, (M + 31)/32);
    dim3 block_dim5(32, 8);
    dim3 grid_dim5((N + 63)/64, (M + 63)/64);
    dim3 block_dim6(16, 16);
    dim3 grid_dim6((N + 127)/128, (M + 127)/128);

    
    /*matmul1<float><<<grid_dim1, block_dim1>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err1) << std::endl;
    }*/


    /*matmul2<float><<<grid_dim2, block_dim2>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err2) << std::endl;
    }*/

    matmul3<float><<<grid_dim3, block_dim3>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err3) << std::endl;
    }

    matmul3_db<float><<<grid_dim3, block_dim3>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err3_db = cudaGetLastError();
    if (err3_db != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err3_db) << std::endl;
    }


    matmul3_4b<float><<<grid_dim3, block_dim3>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err3_4b = cudaGetLastError();
    if (err3_4b != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err3_4b) << std::endl;
    }


    /*matmul4<float><<<grid_dim3, block_dim3>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err4 = cudaGetLastError();
    if (err4 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err4) << std::endl;
    }*/

    /*matmul7<float><<<grid_dim5, block_dim5>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err7 = cudaGetLastError();
    if (err7 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err7) << std::endl;
    }*/

    /*matmul8<<<grid_dim5, block_dim5>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err8 = cudaGetLastError();
    if (err8 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err8) << std::endl;
    }*/


    /*matmul6<float><<<grid_dim6, block_dim6>>>(d_C, d_A, d_B, M, K, N);
    cudaDeviceSynchronize(); 
    cudaError_t err6 = cudaGetLastError();
    if (err6 != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err6) << std::endl;
    }*/


    //cublas对照
    cublasHandle_t handle;
    cublasCreate(&handle);
    matmul_cublas(d_C, d_A, d_B, M, N, K, handle);
    cudaDeviceSynchronize(); 
    cudaError_t err_cublas = cudaGetLastError();
    if (err_cublas != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err_cublas) << std::endl;
    }
    cublasDestroy(handle);
    



    //step4: copy D2H
    cudaMemcpy(h_C, d_C, totalByte_C, cudaMemcpyDeviceToHost);


    //check results


    //step5: cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
