// 和 TMA (GM -> SM) 相关的函数

#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cstdint>


// Get cuTensorMapEncodeTiled driver API.
inline PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = []() {
        cudaDriverEntryPointQueryResult status;
        void* ptr = nullptr;

        CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status
        ));

        if (status != cudaDriverEntryPointSuccess || ptr == nullptr) {
            throw std::runtime_error("cuTensorMapEncodeTiled unavailable");
        }

        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
    }();

    return encode;
}


// Create a 2D BF16 Tensor Map; Global Tensor and Shared Tile are row-major.
inline void encode_tma_2d_bf16(
    CUtensorMap& tensor_map,
    const __nv_bfloat16* data,
    size_t rows,
    size_t cols,
    uint32_t tile_rows,
    uint32_t tile_cols
) {
    if (tile_rows == 0 || tile_cols == 0 ||
        rows < tile_rows || cols < tile_cols ||
        rows % tile_rows != 0 || cols % tile_cols != 0) {
        throw std::runtime_error("Invalid TMA tile dimensions");
    }

    if (reinterpret_cast<uintptr_t>(data) % 16 != 0 ||
        (cols * sizeof(__nv_bfloat16)) % 16 != 0) {
        throw std::runtime_error("Invalid TMA alignment");
    }

    uint64_t global_dim[2] = {cols, rows};
    uint64_t global_stride[1] = {cols * sizeof(__nv_bfloat16)};

    uint32_t box_dim[2] = {tile_cols, tile_rows};
    uint32_t element_stride[2] = {1, 1};

    auto encode = get_cuTensorMapEncodeTiled();

    CUresult err = encode(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<__nv_bfloat16*>(data),
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (err != CUDA_SUCCESS) {
        throw std::runtime_error(
            "cuTensorMapEncodeTiled failed: " +
            std::to_string(static_cast<int>(err))
        );
    }
}


// TMA barrier.
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;


// Select one thread from CTA.
__device__ __forceinline__ bool tma_is_elected() {
    return threadIdx.x == 0;
}


// Copy one 2D tile from Global Memory to Shared Memory.
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const CUtensorMap* tensor_map,
    int32_t coord0,
    int32_t coord1,
    TmaBarrier& barrier
) {
    int32_t coords[2] = {coord0, coord1};

    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        smem_dst,
        tensor_map,
        coords,
        cuda::device::barrier_native_handle(barrier)
    );
}


// One CTA issues two TMA loads and tracks their total transaction bytes.
__device__ __forceinline__ void tma_load_2d_pair(
    void* smem_A,
    const CUtensorMap* tensor_map_A,
    int32_t a_x,
    int32_t a_y,
    void* smem_B,
    const CUtensorMap* tensor_map_B,
    int32_t b_x,
    int32_t b_y,
    TmaBarrier& barrier,
    uint32_t tx_bytes
) {
    if (tma_is_elected()) {
        // Issue two asynchronous TMA loads.
        tma_load_2d(smem_A, tensor_map_A, a_x, a_y, barrier);
        tma_load_2d(smem_B, tensor_map_B, b_x, b_y, barrier);

        // Register expected transaction bytes.
        (void)cuda::device::barrier_arrive_tx(barrier, 1, tx_bytes);
    }
}


// Wait until the specified TMA barrier phase is complete.
__device__ __forceinline__ void tma_wait(
    TmaBarrier& barrier,
    uint32_t phase
) {
    while (!cuda::ptx::mbarrier_try_wait_parity(
        cuda::ptx::sem_acquire,
        cuda::ptx::scope_cta,
        cuda::device::barrier_native_handle(barrier),
        phase
    )) {}
}
