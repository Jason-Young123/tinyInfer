#include "utils.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <climits>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>



static cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = []() {
        cublasHandle_t h;
        cublasStatus_t status = cublasCreate(&h);

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }

        return h;
    }();

    return handle;
}


template <typename T>
void launch_linear_cublas_typed(
    const T* __restrict__ X,       // [M, K], Row Major
    const T* __restrict__ W,       // [N, K], Row Major
    size_t M,
    size_t K,
    size_t N,
    T* __restrict__ O              // [M, N], Row Major
) {
    if (M == 0 || N == 0) {
        return;
    }

    // cuBLAS dimensions use int
    if (M > INT_MAX || K > INT_MAX || N > INT_MAX) {
        throw std::runtime_error("GEMM dimensions exceed INT_MAX");
    }

    const int m = static_cast<int>(M);
    const int n = static_cast<int>(N);
    const int k = static_cast<int>(K);

    // --------------------------------------------------------
    // 1. Data Type
    // --------------------------------------------------------

    cudaDataType_t data_type;
    cublasComputeType_t compute_type;

    if constexpr (std::is_same_v<T, half>) {
        data_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        data_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
    }
    else if constexpr (std::is_same_v<T, float>) {
        data_type = CUDA_R_32F;
        compute_type = CUBLAS_COMPUTE_32F;
    }
    else {
        static_assert(
            std::is_same_v<T, void>,
            "Unsupported GEMM data type"
        );
    }

    // --------------------------------------------------------
    // 2. Scalars
    // --------------------------------------------------------

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // --------------------------------------------------------
    // 3. Get Handle
    // --------------------------------------------------------

    cublasHandle_t handle = get_cublas_handle();

    // --------------------------------------------------------
    // 4. GEMM
    //
    // C = A.T @ B
    //
    // A = W, B = X
    //
    // Physical memory:
    // W: [N, K], Row Major
    // X: [M, K], Row Major
    // O: [M, N], Row Major
    //
    // cuBLAS interpretation:
    // A: [K, N], Column Major
    // B: [K, M], Column Major
    // C: [N, M], Column Major
    // --------------------------------------------------------

    cublasStatus_t status = cublasGemmEx(
        handle,

        // Transpose
        CUBLAS_OP_T,               // A = W -> W
        CUBLAS_OP_N,               // B = X -> X.T

        // Dimensions
        n, m, k,

        // alpha
        &alpha,

        // A = W
        W,
        data_type,
        k,                         // lda

        // B = X
        X,
        data_type,
        k,                         // ldb

        // beta
        &beta,

        // C = O
        O,
        data_type,
        n,                         // ldc

        // Compute Type
        compute_type,

        // Algorithm
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cublasGemmEx failed, status = " +
            std::to_string(static_cast<int>(status))
        );
    }
}



extern "C" int linear_forward_cublas(
    const void* X,
    const void* W,
    size_t M,
    size_t K,
    size_t N,
    void* O,
    int dtype_code
) {
    if (X == nullptr || W == nullptr || O == nullptr ||
        K == 0) {
        return -3;
    }

    try {
        if (dtype_code == 0) {
            launch_linear_cublas_typed<half>(
                static_cast<const half*>(X),
                static_cast<const half*>(W),
                M, K, N,
                static_cast<half*>(O)
            );
            return 0;
        }

        if (dtype_code == 1) {
            launch_linear_cublas_typed<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const __nv_bfloat16*>(W),
                M, K, N,
                static_cast<__nv_bfloat16*>(O)
            );
            return 0;
        }

        if (dtype_code == 2) {
            launch_linear_cublas_typed<float>(
                static_cast<const float*>(X),
                static_cast<const float*>(W),
                M, K, N,
                static_cast<float*>(O)
            );
            return 0;
        }

        return -1;
    }
    catch (...) {
        return -2;
    }
}



/************************************* cutlass ***********************************/
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// CTA Tile; 一个 CTA/block 计算多大的输出块
using CTA_Tile = cutlass::gemm::GemmShape<128, 128, 32>;
// Warp Tile; 一个 Warp 负责多大的计算块
using Warp_Tile = cutlass::gemm::GemmShape<64, 64, 32>;
// Tensor Core MMA Tile; Tensor Core MMA 指令的计算粒度
using MMA_Tile = cutlass::gemm::GemmShape<16, 8, 16>;
// Pipeline
constexpr int NUM_STAGES = 3;

// CUDA type -> CUTLASS type
template <typename T>
struct CutlassType;
template <>
struct CutlassType<half> { using type = cutlass::half_t; };
template <>
struct CutlassType<__nv_bfloat16> { using type = cutlass::bfloat16_t; };


// 实现无bias的线性层计算(cutlass版本); 目前type支持half/bf16
// O = X * W.T
template <typename T>
void launch_linear_cutlass_typed(
    const T* __restrict__ X,   // [M, K], Row Major
    const T* __restrict__ W,   // [N, K], Row Major
    size_t M,
    size_t K,
    size_t N,
    T* __restrict__ O          // [M, N], Row Major
) {
    using ElementType = typename CutlassType<T>::type; // 将T(cuda type)转为cutlass type
    constexpr int ElementsPerAccess = 8;
    // D = alpha * accumulator + beta * C
    using Epilogue = cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerAccess, Accumulator, float>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementType, LayoutA,
        ElementType, LayoutB,
        ElementType, LayoutC,
        Accumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        CTA_Tile,
        Warp_Tile,
        MMA_Tile,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NUM_STAGES,
        ElementsPerAccess,          // 从A读取数据时的向量化粒度(一次加载多少个数据)
        ElementsPerAccess           // 从B读取数据时的向量化粒度(一次加载多少个数据)
    >;

    auto A = reinterpret_cast<const ElementType*>(X);
    auto B = reinterpret_cast<const ElementType*>(W);
    auto C = reinterpret_cast<ElementType*>(O);

    Gemm gemm;
    typename Gemm::Arguments args(
        {int(M), int(N), int(K)},
        {A, int(K)},    // A = X [M, K], Row Major
        {B, int(K)},    // B = W [K, N], Column Major
        {C, int(N)},    // C = O [M, N], Row Major
        {C, int(N)},    // D = O [M, N], Row Major
        {1.0f, 0.0f}    // alpha, beta
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm(args, nullptr, stream);
}




extern "C" void linear_forward_cutlass_(
    const void* X,
    const void* W,
    size_t M,
    size_t K,
    size_t N,
    void* O,
    int dtype_code
) {
    if (dtype_code == 0) { // half
        launch_linear_cutlass_typed<half>(
            static_cast<const half*>(X),
            static_cast<const half*>(W),
            M, K, N,
            static_cast<half*>(O)
        );
        return;
    }

    if (dtype_code == 1) { // bf16
        launch_linear_cutlass_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(X),
            static_cast<const __nv_bfloat16*>(W),
            M, K, N,
            static_cast<__nv_bfloat16*>(O)
        );
        return;
    }

    // 暂时未实现float
    throw std::runtime_error("unsupported dtype_code");

}
