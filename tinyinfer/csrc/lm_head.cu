#include "utils.cuh"
#include "cpasync.cuh"
#include "mma.cuh"
#include "tma.cuh"
#include <mma.h>
#include <cuda_bf16.h>

using namespace nvcuda;


constexpr int LM_HEAD_THREADS = 128;
constexpr int LM_HEAD_NUM_WARPS = LM_HEAD_THREADS / 32;



/************************************** gemv ************************************/
// 一个 Warp 负责一个 Vocabulary Logit; 针对GEMV或者小M GEMM
// Grid = [ceil(vocab_size / 8), M]
template <typename T, int VEC_SIZE>
__global__ void lm_head_gemv(
    const T* __restrict__ X,       // [M, hidden_size]
    const T* __restrict__ weight,  // [vocab_size, hidden_size]
    size_t hidden_size,
    size_t vocab_size,
    T* __restrict__ O              // [M, vocab_size]
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int warpid = tid >> 5;
    int laneid = tid & 31;

    float sum = 0.f;

    size_t vocab_idx = static_cast<size_t>(bidx) * LM_HEAD_NUM_WARPS + warpid;
    if (vocab_idx >= vocab_size) return;

    size_t base_x = static_cast<size_t>(bidy) * hidden_size;
    size_t base_w = vocab_idx * hidden_size;

    const float4* X4 = reinterpret_cast<const float4*>(X + base_x);
    const float4* W4 = reinterpret_cast<const float4*>(weight + base_w);
    size_t num_vec = hidden_size / VEC_SIZE;

    // Prologue
    float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 x0 = zero;
    float4 w0 = zero;
    size_t idx0 = laneid;
    if (idx0 < num_vec) {
        x0 = X4[idx0];
        w0 = W4[idx0];
    }

    // steady state
    size_t num_iters = (num_vec + warpSize - 1) / warpSize;
    for (size_t iter = 1; iter < num_iters; ++iter) {
        size_t idx = laneid + iter * warpSize;
        float4 x1 = zero;
        float4 w1 = zero;
        if (idx < num_vec) {
            x1 = X4[idx];
            w1 = W4[idx];
        }

        const T* xptr = reinterpret_cast<const T*>(&x0);
        const T* wptr = reinterpret_cast<const T*>(&w0);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            sum = fmaf(to_float(xptr[j]), to_float(wptr[j]), sum);
        }

        x0 = x1;
        w0 = w1;
    }

    // Epilogue
    const T* xptr = reinterpret_cast<const T*>(&x0);
    const T* wptr = reinterpret_cast<const T*>(&w0);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
        sum = fmaf(to_float(xptr[j]), to_float(wptr[j]), sum);
    }

    sum = warp_reduce_sum<float>(sum);

    if (laneid == 0) {
        O[bidy * vocab_size + vocab_idx] = from_float<T>(sum);
    }
}

template <typename T, int VEC_SIZE>
void launch_lm_head_gemv_typed(
    const T* X,
    const T* weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    T* O
) {
    if (hidden_size % VEC_SIZE != 0) {
        throw std::runtime_error("Invalid LM Head dimensions");
    }

    const dim3 block_dim(LM_HEAD_THREADS);
    const dim3 grid_dim((vocab_size + LM_HEAD_NUM_WARPS - 1) / LM_HEAD_NUM_WARPS, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lm_head_gemv<T, VEC_SIZE><<<grid_dim, block_dim, 0, stream>>>(X, weight, hidden_size, vocab_size, O);
    CUDA_CHECK(cudaGetLastError());
}




/*********************************** gemm_ccore *************************************/
template <typename T, int VEC_SIZE, int M_TILE>
__global__ void lm_head_gemm_ccore(
    const T* __restrict__ X,
    const T* __restrict__ weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    T* __restrict__ O
) {
    static_assert(sizeof(T) * VEC_SIZE == 16);

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t vocab_idx = static_cast<size_t>(bidx) * LM_HEAD_NUM_WARPS + warpid;
    if (vocab_idx >= vocab_size) return;

    size_t base_w = vocab_idx * hidden_size;

    const float4* W4 = reinterpret_cast<const float4*>(weight + base_w);

    size_t num_vec = hidden_size / VEC_SIZE;
    size_t num_iters = (num_vec + warpSize - 1) / warpSize;

    float sum[M_TILE] = {0.f};

    #pragma unroll 1
    for (size_t iter = 0; iter < num_iters; ++iter) {
        size_t idx = laneid + iter * warpSize;

        float4 w = make_float4(0.f, 0.f, 0.f, 0.f);
        if (idx < num_vec) {
            w = W4[idx];
        }
        const T* wptr = reinterpret_cast<const T*>(&w);

        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {//每个warp负责1个vocab的M_TILE个token
            size_t token_idx = static_cast<size_t>(bidy) * M_TILE + m;
            if (token_idx < M && idx < num_vec) {
                const float4* X4 = reinterpret_cast<const float4*>(X + token_idx * hidden_size);
                float4 x = X4[idx];
                const T* xptr = reinterpret_cast<const T*>(&x);
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; ++j) {
                    sum[m] = fmaf(to_float(xptr[j]), to_float(wptr[j]), sum[m]);
                }
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
void launch_lm_head_gemm_ccore_typed(
    const T* X,
    const T* weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    T* O
) {
    const dim3 block_dim(LM_HEAD_THREADS);
    const dim3 grid_dim((vocab_size + LM_HEAD_NUM_WARPS - 1) / LM_HEAD_NUM_WARPS, (M + M_TILE - 1) / M_TILE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lm_head_gemm_ccore<T, VEC_SIZE, M_TILE><<<grid_dim, block_dim, 0, stream>>>(X, weight, M, hidden_size, vocab_size, O);

    CUDA_CHECK(cudaGetLastError());
}




/***************************** gemm_tcore: cp.async + wmma + wmma *****************************/
/*
// WMMA_* 为固定参数
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;


// One Warp computes one (M_TILE x N_TILE) output tile.
// One CTA contains 4 independent Warps, producing a M_TILE x (4 * N_TILE) output tile.
template <int M_TILE, int K_TILE, int N_TILE, int PP_STAGE>
__global__ void lm_head_gemm_tcore(
    const __nv_bfloat16* __restrict__ X,       // [M, hidden_size]; A
    const __nv_bfloat16* __restrict__ weight,  // [vocab_size, hidden_size]; B
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    __nv_bfloat16* __restrict__ O              // [M, vocab_size]
) {
    // Compile-time parameters, 由cpu在编译期计算得到
    constexpr int K_STRIDE_BYTES = get_min_stride_bytes(K_TILE * sizeof(__nv_bfloat16), 4, 4);
    constexpr int N_STRIDE_BYTES = get_min_stride_bytes(WMMA_N * sizeof(float), 4, 8);
    static_assert(K_STRIDE_BYTES > 0 && N_STRIDE_BYTES > 0, "Invalid shared memory stride");
    static_assert(K_STRIDE_BYTES % 16 == 0 && N_STRIDE_BYTES % 16 == 0, "Invalid WMMA stride alignment");

    constexpr int K_STRIDE = K_STRIDE_BYTES / sizeof(__nv_bfloat16);
    constexpr int N_STRIDE = N_STRIDE_BYTES / sizeof(float);
    constexpr int K_PADDING = K_STRIDE - K_TILE;
    constexpr int M_TILES = M_TILE / WMMA_M;
    constexpr int N_TILES = N_TILE / WMMA_N;

    static_assert(M_TILE > 0 && K_TILE > 0 && N_TILE > 0, "Invalid tile size");
    static_assert((M_TILE % WMMA_M == 0) && (K_TILE % WMMA_K == 0) && (N_TILE % WMMA_N == 0), "Invalid M_TILE/K_TILE/N_TILE");
    static_assert(PP_STAGE >= 2 && PP_STAGE <= 9, "Invalid PP_STAGE");

    // A/B tile in shared memory; multi-stage buffering.
    struct SharedAB {
        __nv_bfloat16 sA[PP_STAGE][M_TILE * K_STRIDE];
        __nv_bfloat16 sB[PP_STAGE][LM_HEAD_NUM_WARPS * N_TILE * K_STRIDE];
    };

    union SharedStorage {
        SharedAB ab;
        // Temporary output tile for converting FP32 accumulators to BF16.
        float sC[M_TILES][LM_HEAD_NUM_WARPS * N_TILES * WMMA_M * N_STRIDE];
    };

    int tid = threadIdx.x;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t K = hidden_size, N = vocab_size;
    size_t n0 = static_cast<size_t>(blockIdx.y) * LM_HEAD_NUM_WARPS * N_TILE;
    size_t m0 = static_cast<size_t>(blockIdx.x) * M_TILE;

    __shared__ __align__(32) SharedStorage smem;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[M_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[N_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[M_TILES][N_TILES];

    #pragma unroll
    for (int i = 0; i < M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    // K dimension: ceil(K / K_TILE) iterations.
    size_t num_iters = (K + K_TILE - 1) / K_TILE;

    // Prologue: Prefetch the first PP_STAGE - 1 tiles.
    for (int stage = 0; stage < PP_STAGE - 1 && static_cast<size_t>(stage) < num_iters; ++stage) {
        size_t k0 = static_cast<size_t>(stage) * K_TILE;
        cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sA[stage][0], X, M, K, m0, k0, M_TILE, K_TILE, K_PADDING);
        cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sB[stage][0], weight, N, K, n0, k0, LM_HEAD_NUM_WARPS * N_TILE, K_TILE, K_PADDING);
        commit_group();
    }

    // Wait until the first tile is ready.
    if (num_iters < PP_STAGE - 1) wait_group<0>();
    else wait_group<PP_STAGE - 2>();
    __syncthreads();

    // Mainloop: PP_STAGE-stage pipeline.
    for (size_t iter = 0; iter < num_iters; ++iter) {
        int curr = static_cast<int>(iter % PP_STAGE);
        size_t prefetch_iter = iter + PP_STAGE - 1;
        int next = static_cast<int>(prefetch_iter % PP_STAGE);

        // Stage 1: Prefetch the NEXT tile asynchronously.
        if (prefetch_iter < num_iters) {
            size_t next_k0 = prefetch_iter * K_TILE;
            cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sA[next][0], X, M, K, m0, next_k0, M_TILE, K_TILE, K_PADDING);
            cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sB[next][0], weight, N, K, n0, next_k0, LM_HEAD_NUM_WARPS * N_TILE, K_TILE, K_PADDING);
            commit_group();
        }

        // Stage 2: Load CURRENT tile and perform Tensor Core MMA.
        #pragma unroll
        for (int k = 0; k < K_TILE; k += WMMA_K) {
            #pragma unroll
            for (int j = 0; j < N_TILES; ++j) { // Load B Fragments.
                wmma::load_matrix_sync(b_frag[j], &smem.ab.sB[curr][(warpid * N_TILE + j * WMMA_N) * K_STRIDE + k], K_STRIDE);
            }
            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) { // Load A Fragments.
                wmma::load_matrix_sync(a_frag[i], &smem.ab.sA[curr][i * WMMA_M * K_STRIDE + k], K_STRIDE);
            }

            // Stage 3: Tensor Core MMA.
            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < N_TILES; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        // Stage 4: Wait for the NEXT tile.
        if (iter + 1 < num_iters) {
            if (prefetch_iter < num_iters) wait_group<PP_STAGE - 2>();
            else wait_group<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // Store FP32 accumulators to shared memory.
    #pragma unroll
    for (int i = 0; i < M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
            wmma::store_matrix_sync(
                &smem.sC[i][(warpid * N_TILES + j) * WMMA_M * N_STRIDE],
                c_frag[i][j],
                N_STRIDE,
                wmma::mem_row_major
            );
        }
    }
    __syncwarp();

    // Convert FP32 output to BF16.
    #pragma unroll
    for (int tile_m = 0; tile_m < M_TILES; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < N_TILES; ++tile_n) {
            #pragma unroll
            for (int i = laneid; i < WMMA_M * WMMA_N; i += 32) {
                int _m = i / WMMA_N;
                int _n = i % WMMA_N;
                size_t m = m0 + tile_m * WMMA_M + _m;
                size_t n = n0 + warpid * N_TILE + tile_n * WMMA_N + _n;

                if (m < M && n < N) {
                    O[m * N + n] = __float2bfloat16_rn(
                        smem.sC[tile_m][(warpid * N_TILES + tile_n) * WMMA_M * N_STRIDE + _m * N_STRIDE + _n]
                    );
                }
            }
        }
    }
}
*/







/******************************* gemm_tcore: cp.async + ldmatrix + mma *******************************/
/*
// WMMA_* 为固定参数
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// One Warp computes one (M_TILE x N_TILE) output tile.
// One CTA contains 4 independent Warps, producing a M_TILE x (4 * N_TILE) output tile.
template <int M_TILE, int K_TILE, int N_TILE, int PP_STAGE>
__global__ void lm_head_gemm_tcore(
    const __nv_bfloat16* __restrict__ X,       // [M, hidden_size]; A
    const __nv_bfloat16* __restrict__ weight,  // [vocab_size, hidden_size]; B
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    __nv_bfloat16* __restrict__ O              // [M, vocab_size]
) {
    // Compile-time parameters, 由cpu在编译期计算得到
    constexpr int K_STRIDE_BYTES = get_min_stride_bytes(K_TILE * sizeof(__nv_bfloat16), 4, 4);
    constexpr int N_STRIDE_BYTES = get_min_stride_bytes(WMMA_N * sizeof(float), 4, 8);
    static_assert(K_STRIDE_BYTES > 0 && N_STRIDE_BYTES > 0, "Invalid shared memory stride");
    static_assert(K_STRIDE_BYTES % 16 == 0 && N_STRIDE_BYTES % 16 == 0, "Invalid WMMA stride alignment");

    constexpr int K_STRIDE = K_STRIDE_BYTES / sizeof(__nv_bfloat16);
    constexpr int N_STRIDE = N_STRIDE_BYTES / sizeof(float);
    constexpr int K_PADDING = K_STRIDE - K_TILE;
    constexpr int M_TILES = M_TILE / WMMA_M;
    constexpr int N_TILES = N_TILE / WMMA_N;

    // [MOD] mma.sync.m16n8k16, N方向每16列拆分为2个8列Tile
    constexpr int MMA_N = 8;
    constexpr int N_MMA_TILES = N_TILE / MMA_N;

    static_assert(M_TILE > 0 && K_TILE > 0 && N_TILE > 0, "Invalid tile size");
    static_assert((M_TILE % WMMA_M == 0) && (K_TILE % WMMA_K == 0) && (N_TILE % WMMA_N == 0), "Invalid M_TILE/K_TILE/N_TILE");
    static_assert(PP_STAGE >= 2 && PP_STAGE <= 9, "Invalid PP_STAGE");

    // A/B tile in shared memory; multi-stage buffering.
    struct SharedAB {
        __nv_bfloat16 sA[PP_STAGE][M_TILE * K_STRIDE];
        __nv_bfloat16 sB[PP_STAGE][LM_HEAD_NUM_WARPS * N_TILE * K_STRIDE];
    };

    union SharedStorage {
        SharedAB ab;
        // Temporary output tile for converting FP32 accumulators to BF16.
        float sC[M_TILES][LM_HEAD_NUM_WARPS * N_TILES * WMMA_M * N_STRIDE];
    };

    int tid = threadIdx.x;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t K = hidden_size, N = vocab_size;
    size_t n0 = static_cast<size_t>(blockIdx.x) * LM_HEAD_NUM_WARPS * N_TILE;
    size_t m0 = static_cast<size_t>(blockIdx.y) * M_TILE;

    __shared__ __align__(32) SharedStorage smem;

    // [MOD] 显式管理每个Lane的A/B Fragment和FP32 Accumulator
    uint32_t a_frag[M_TILES][4];
    uint32_t b_frag[N_MMA_TILES][2];
    float c_frag[M_TILES][N_MMA_TILES][4] = {};

    // K dimension: ceil(K / K_TILE) iterations.
    size_t num_iters = (K + K_TILE - 1) / K_TILE;

    // Prologue: Prefetch the first PP_STAGE - 1 tiles.
    for (int stage = 0; stage < PP_STAGE - 1 && static_cast<size_t>(stage) < num_iters; ++stage) {
        size_t k0 = static_cast<size_t>(stage) * K_TILE;
        cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sA[stage][0], X, M, K, m0, k0, M_TILE, K_TILE, K_PADDING);
        cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sB[stage][0], weight, N, K, n0, k0, LM_HEAD_NUM_WARPS * N_TILE, K_TILE, K_PADDING);
        commit_group();
    }

    // Wait until the first tile is ready.
    if (num_iters < PP_STAGE - 1) wait_group<0>();
    else wait_group<PP_STAGE - 2>();
    __syncthreads();

    // Mainloop: PP_STAGE-stage pipeline.
    for (size_t iter = 0; iter < num_iters; ++iter) {
        int curr = static_cast<int>(iter % PP_STAGE);
        size_t prefetch_iter = iter + PP_STAGE - 1;
        int next = static_cast<int>(prefetch_iter % PP_STAGE);

        // Stage 1: Prefetch the NEXT tile asynchronously.
        if (prefetch_iter < num_iters) {
            size_t next_k0 = prefetch_iter * K_TILE;
            cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sA[next][0], X, M, K, m0, next_k0, M_TILE, K_TILE, K_PADDING);
            cooperative_load_2d<__nv_bfloat16, 16>(&smem.ab.sB[next][0], weight, N, K, n0, next_k0, LM_HEAD_NUM_WARPS * N_TILE, K_TILE, K_PADDING);
            commit_group();
        }

        // Stage 2: Load CURRENT tile and perform Tensor Core MMA.
        #pragma unroll
        for (int k = 0; k < K_TILE; k += WMMA_K) {
            // [MOD] Load B Fragments, 每个8列Tile使用ldmatrix.x2
            #pragma unroll
            for (int j = 0; j < N_MMA_TILES; ++j) {
                int b_row = warpid * N_TILE + j * MMA_N + (laneid & 7);
                int b_col = k + ((laneid >> 3) & 1) * 8;
                ldmatrix_b2(b_frag[j], &smem.ab.sB[curr][b_row * K_STRIDE + b_col]);
            }

            // [MOD] Load A Fragments, 每个16行Tile使用ldmatrix.x4
            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) {
                int a_row = i * WMMA_M + (laneid & 7) + ((laneid >> 3) & 1) * 8;
                int a_col = k + (laneid >> 4) * 8;
                ldmatrix_a4(a_frag[i], &smem.ab.sA[curr][a_row * K_STRIDE + a_col]);
            }

            // Stage 3: Tensor Core MMA.
            // [MOD] 每次mma.sync计算16x8x16
            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < N_MMA_TILES; ++j) {
                    mma_bf16(c_frag[i][j], a_frag[i], b_frag[j]);
                }
            }
        }

        // Stage 4: Wait for the NEXT tile.
        if (iter + 1 < num_iters) {
            if (prefetch_iter < num_iters) wait_group<PP_STAGE - 2>();
            else wait_group<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // [MOD] Direct register-to-global epilogue, 无需经过Shared Memory中转
    #pragma unroll
    for (int tile_m = 0; tile_m < M_TILES; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < N_MMA_TILES; ++tile_n) {
            int group = laneid >> 2;
            int t = laneid & 3;

            size_t m0_out = m0 + tile_m * WMMA_M + group;
            size_t m1_out = m0_out + 8;
            size_t n_out = n0 + warpid * N_TILE + tile_n * MMA_N + t * 2;

            if (m0_out < M) {
                if (n_out < N) O[m0_out * N + n_out] = __float2bfloat16_rn(c_frag[tile_m][tile_n][0]);
                if (n_out + 1 < N) O[m0_out * N + n_out + 1] = __float2bfloat16_rn(c_frag[tile_m][tile_n][1]);
            }

            if (m1_out < M) {
                if (n_out < N) O[m1_out * N + n_out] = __float2bfloat16_rn(c_frag[tile_m][tile_n][2]);
                if (n_out + 1 < N) O[m1_out * N + n_out + 1] = __float2bfloat16_rn(c_frag[tile_m][tile_n][3]);
            }
        }
    }
}*/





/******************************* gemm_tcore: tma + wmma + wmma *******************************/
// WMMA_* 为固定参数
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;


// One Warp computes one (M_TILE x N_TILE) output tile.
// One CTA contains 4 independent Warps, producing a M_TILE x (4 * N_TILE) output tile.
template <int M_TILE, int K_TILE, int N_TILE, int PP_STAGE>
__global__ void lm_head_gemm_tcore(
    const __grid_constant__ CUtensorMap tensor_map_A,
    const __grid_constant__ CUtensorMap tensor_map_B,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    __nv_bfloat16* __restrict__ O
) {
    // Compile-time parameters.
    constexpr int K_STRIDE = K_TILE;
    constexpr int N_STRIDE_BYTES = get_min_stride_bytes(WMMA_N * sizeof(float), 4, 8);

    static_assert(N_STRIDE_BYTES > 0 && N_STRIDE_BYTES % 16 == 0,
                  "Invalid WMMA output stride");

    constexpr int N_STRIDE = N_STRIDE_BYTES / sizeof(float);
    constexpr int M_TILES = M_TILE / WMMA_M;
    constexpr int N_TILES = N_TILE / WMMA_N;

    constexpr uint32_t BYTES_A =
        M_TILE * K_TILE * sizeof(__nv_bfloat16);

    constexpr uint32_t BYTES_B =
        LM_HEAD_NUM_WARPS * N_TILE * K_TILE * sizeof(__nv_bfloat16);

    constexpr uint32_t TMA_BYTES = BYTES_A + BYTES_B;

    static_assert(M_TILE > 0 && K_TILE > 0 && N_TILE > 0,
                  "Invalid tile size");
    static_assert(M_TILE % WMMA_M == 0, "Invalid M_TILE");
    static_assert(K_TILE % WMMA_K == 0, "Invalid K_TILE");
    static_assert(N_TILE % WMMA_N == 0, "Invalid N_TILE");
    static_assert(PP_STAGE >= 2, "Invalid PP_STAGE");

    static_assert(BYTES_A % 128 == 0 && BYTES_B % 128 == 0,
                  "Invalid TMA shared memory alignment");

    // A/B tile in shared memory; continuous row-major layout.
    struct SharedAB {
        alignas(128) __nv_bfloat16 sA[PP_STAGE][M_TILE * K_STRIDE];
        alignas(128) __nv_bfloat16 sB[PP_STAGE][LM_HEAD_NUM_WARPS * N_TILE * K_STRIDE];
        alignas(8) TmaBarrier barrier[PP_STAGE];
    };

    union SharedStorage {
        SharedAB ab;
        // Temporary output tile for converting FP32 accumulators to BF16.
        float sC[M_TILES][LM_HEAD_NUM_WARPS * N_TILES * WMMA_M * N_STRIDE];
    };

    int tid = threadIdx.x;
    int warpid = tid >> 5;
    int laneid = tid & 31;

    size_t K = hidden_size, N = vocab_size;

    // CTA Swizzling: blockIdx.x -> M, blockIdx.y -> N.
    size_t m0 = static_cast<size_t>(blockIdx.x) * M_TILE;
    size_t n0 = static_cast<size_t>(blockIdx.y) * LM_HEAD_NUM_WARPS * N_TILE;

    __shared__ __align__(128) SharedStorage smem;

    // Initialize one transaction-based barrier per stage.
    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < PP_STAGE; ++stage) {
            cuda::ptx::mbarrier_init(&smem.ab.barrier[stage], 1);
        }

        // Make barrier initialization visible to the async proxy.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    uint32_t barrier_phase[PP_STAGE] = {};

    // WMMA Fragments and FP32 Accumulators.
    wmma::fragment<wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        __nv_bfloat16, wmma::row_major> a_frag[M_TILES];

    wmma::fragment<wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        __nv_bfloat16, wmma::col_major> b_frag[N_TILES];

    wmma::fragment<wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float> c_frag[M_TILES][N_TILES];

    #pragma unroll
    for (int i = 0; i < M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // K dimension: K must be divisible by K_TILE.
    size_t num_iters = K / K_TILE;

    // Prologue: Prefetch the first PP_STAGE - 1 tiles.
    for (int stage = 0;
         stage < PP_STAGE - 1 && static_cast<size_t>(stage) < num_iters;
         ++stage) {

        size_t k0 = static_cast<size_t>(stage) * K_TILE;

        tma_load_2d_pair(
            &smem.ab.sA[stage][0], &tensor_map_A,
            static_cast<int32_t>(k0), static_cast<int32_t>(m0),
            &smem.ab.sB[stage][0], &tensor_map_B,
            static_cast<int32_t>(k0), static_cast<int32_t>(n0),
            &smem.ab.barrier[stage], TMA_BYTES
        );
    }
    __syncthreads();

    // Mainloop: PP_STAGE-stage TMA pipeline.
    for (size_t iter = 0; iter < num_iters; ++iter) {
        int curr = static_cast<int>(iter % PP_STAGE);

        // Stage 1: Wait for CURRENT A/B tiles.
        tma_wait(&smem.ab.barrier[curr], barrier_phase[curr]);
        barrier_phase[curr] ^= 1;

        size_t prefetch_iter = iter + PP_STAGE - 1;
        int next = static_cast<int>(prefetch_iter % PP_STAGE);

        // Stage 2: Prefetch NEXT A/B tiles asynchronously.
        if (prefetch_iter < num_iters) {
            size_t next_k0 = prefetch_iter * K_TILE;

            tma_load_2d_pair(
                &smem.ab.sA[next][0], &tensor_map_A,
                static_cast<int32_t>(next_k0), static_cast<int32_t>(m0),
                &smem.ab.sB[next][0], &tensor_map_B,
                static_cast<int32_t>(next_k0), static_cast<int32_t>(n0),
                &smem.ab.barrier[next], TMA_BYTES
            );

            // Ensure TMA has been issued before proceeding.
            // This does not wait for the actual data transfer.
            __syncthreads();
        }

        // Stage 3: Original WMMA computation.
        #pragma unroll
        for (int k = 0; k < K_TILE; k += WMMA_K) {

            #pragma unroll
            for (int j = 0; j < N_TILES; ++j) {
                // Load B Fragments.
                wmma::load_matrix_sync(
                    b_frag[j],
                    &smem.ab.sB[curr][
                        (warpid * N_TILE + j * WMMA_N) * K_STRIDE + k
                    ],
                    K_STRIDE
                );
            }

            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) {
                // Load A Fragments.
                wmma::load_matrix_sync(
                    a_frag[i],
                    &smem.ab.sA[curr][
                        i * WMMA_M * K_STRIDE + k
                    ],
                    K_STRIDE
                );
            }

            // Tensor Core MMA.
            #pragma unroll
            for (int i = 0; i < M_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < N_TILES; ++j) {
                    wmma::mma_sync(
                        c_frag[i][j],
                        a_frag[i],
                        b_frag[j],
                        c_frag[i][j]
                    );
                }
            }
        }

        // All consumers must finish reading CURRENT SMEM before stage reuse.
        if (iter + 1 < num_iters) {
            __syncthreads();
        }
    }
    __syncthreads();

    // Store FP32 accumulators to shared memory.
    #pragma unroll
    for (int i = 0; i < M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
            wmma::store_matrix_sync(
                &smem.sC[i][(warpid * N_TILES + j) * WMMA_M * N_STRIDE],
                c_frag[i][j],
                N_STRIDE,
                wmma::mem_row_major
            );
        }
    }
    __syncwarp();

    // Convert FP32 output to BF16.
    #pragma unroll
    for (int tile_m = 0; tile_m < M_TILES; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < N_TILES; ++tile_n) {
            #pragma unroll
            for (int i = laneid; i < WMMA_M * WMMA_N; i += 32) {
                int _m = i / WMMA_N;
                int _n = i % WMMA_N;

                size_t m = m0 + tile_m * WMMA_M + _m;
                size_t n = n0 + warpid * N_TILE + tile_n * WMMA_N + _n;

                if (m < M && n < N) {
                    O[m * N + n] = __float2bfloat16_rn(
                        smem.sC[tile_m][
                            (warpid * N_TILES + tile_n) * WMMA_M * N_STRIDE
                            + _m * N_STRIDE + _n
                        ]
                    );
                }
            }
        }
    }
}








template <int M_TILE, int K_TILE, int N_TILE, int PP_STAGE>
void launch_lm_head_gemm_tcore_tma(
    const __nv_bfloat16* X,
    const __nv_bfloat16* weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    __nv_bfloat16* O
) {
    if (M % M_TILE != 0 ||
        hidden_size % K_TILE != 0 ||
        vocab_size % (LM_HEAD_NUM_WARPS * N_TILE) != 0) {
        throw std::runtime_error("TMA requires full tiles without OOB");
    }

    // Create A/B Tensor Maps.
    CUtensorMap tensor_map_A{};
    CUtensorMap tensor_map_B{};

    encode_tma_2d_bf16(
        tensor_map_A, X,
        M, hidden_size,
        M_TILE, K_TILE
    );

    encode_tma_2d_bf16(
        tensor_map_B, weight,
        vocab_size, hidden_size,
        LM_HEAD_NUM_WARPS * N_TILE, K_TILE
    );

    const dim3 block_dim(LM_HEAD_THREADS);
    const dim3 grid_dim(
        M / M_TILE,
        vocab_size / (LM_HEAD_NUM_WARPS * N_TILE)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    lm_head_gemm_tcore<M_TILE, K_TILE, N_TILE, PP_STAGE>
        <<<grid_dim, block_dim, 0, stream>>>(
            tensor_map_A,
            tensor_map_B,
            M, hidden_size, vocab_size, O
        );

    CUDA_CHECK(cudaGetLastError());
}




template <int M_TILE, int K_TILE, int N_TILE, int PP_STAGE>
void launch_lm_head_gemm_tcore(
    const __nv_bfloat16* X,
    const __nv_bfloat16* weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    __nv_bfloat16* O
) {
    const dim3 block_dim(LM_HEAD_THREADS);
    /*const dim3 grid_dim(
        (vocab_size + LM_HEAD_NUM_WARPS * N_TILE - 1) / (LM_HEAD_NUM_WARPS * N_TILE),
        (M + M_TILE - 1) / M_TILE
    );*/
    const dim3 grid_dim(
        (M + M_TILE - 1) / M_TILE,
        (vocab_size + LM_HEAD_NUM_WARPS * N_TILE - 1) / (LM_HEAD_NUM_WARPS * N_TILE)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lm_head_gemm_tcore<M_TILE, K_TILE, N_TILE, PP_STAGE><<<grid_dim, block_dim, 0, stream>>>(X, weight, M, hidden_size, vocab_size, O);
    CUDA_CHECK(cudaGetLastError());
}










extern "C" void lm_head_forward_cuda(
    const void* X,
    const void* weight,
    size_t M,
    size_t hidden_size,
    size_t vocab_size,
    void* O,
    int dtype_code
) {
    if (dtype_code != 1) {
        throw std::runtime_error("LM Head only supports BF16");
    }

    const auto* x = static_cast<const __nv_bfloat16*>(X);
    const auto* w = static_cast<const __nv_bfloat16*>(weight);
    auto* o = static_cast<__nv_bfloat16*>(O);


    if (M <= 2) {
        launch_lm_head_gemm_ccore_typed<__nv_bfloat16, 8, 2>(
            x, w, M, hidden_size, vocab_size, o
        );
    }
    else if (M <= 4) {
        launch_lm_head_gemm_ccore_typed<__nv_bfloat16, 8, 4>(
            x, w, M, hidden_size, vocab_size, o
        );
    }
    else if (M <= 6) {
        launch_lm_head_gemm_ccore_typed<__nv_bfloat16, 8, 8>(
            x, w, M, hidden_size, vocab_size, o
        );
    }
    else if (M <= 16) {
        launch_lm_head_gemm_tcore<16, 64, 16, 2>(
            x, w, M, hidden_size, vocab_size, o
        );
    }
    else{ 
        launch_lm_head_gemm_tcore_tma<32, 64, 16, 2>(
            x, w, M, hidden_size, vocab_size, o
        );
    }
}