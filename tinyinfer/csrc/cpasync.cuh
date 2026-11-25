// 和 cp.async (GM -> SM)相关的函数

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>


// Copy from Global Memory to Shared Memory asynchronously.
// 注意cp.async的粒度仅(Byte)支持4B/8B/16B
// Byte代表写入GMEM的总字节数, 而src_size代表从GMEM读取的总字节数; 二者满足 0 <= src_size <= Byte
// GMEM用64bit地址空间, 而SMEM只需32bit地址空间
template <size_t Byte>
__device__ __forceinline__ void cp_async(
    void* smem_ptr,
    const void* gmem_ptr,
    int src_size
) {
    static_assert(Byte == 4 || Byte == 8 || Byte == 16, "Unsupported cp.async size");
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr),
          "n"(Byte),
          "r"(src_size)
    );
}



// Commit all previously issued cp.async operations into a group.
__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}



// Wait until at most N groups remain pending.
template <int N>
__device__ __forceinline__ void wait_group() {
    static_assert(N >= 0 && N <= 7, "Invalid wait_group value");
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}



// 1个CTA协作加载1d数据; 一共加载num_elements个元素, 每个线程每次加载Byte字节
template <typename T, size_t Byte>
__device__ __forceinline__ void cooperative_load_1d(
    T* __restrict__ smem_dst,
    const T* __restrict__ gmem_src,
    size_t num_elements
) {
    static_assert(Byte == 4 || Byte == 8 || Byte == 16, "Unsupported cp.async size");
    static_assert(Byte % sizeof(T) == 0, "Invalid cooperative load config");
    //static_assert(num_elements % vec_size == 0, "");
    
    int tid = threadIdx.x;
    size_t vec_size = Byte / sizeof(T);
    size_t vec_num = num_elements / vec_size;

    // Stage 1: CTA cooperatively loads full vectors.
    #pragma unroll
    for (size_t i = tid; i < vec_num; i += blockDim.x) {
        cp_async<Byte>(smem_dst + i * vec_size, gmem_src + i * vec_size, Byte);
    }

    // Stage 2: Handle remaining elements.
    //if (tid == 0) {
    //    for (size_t i = vec_num * vec_size; i < num_elements; ++i) {
    //        smem_dst[i] = gmem_src[i];
    //    }
    //}
}




// 1个CTA协作加载2d数据; 一共加载row * col个元素, 每个线程每次加载Byte字节
// 被调用时必须检查确保col % (Byte/sizeof(T)) == 0, 即加载时允许整体偏移非VEC_SIZE整数倍、越界等特殊情况, 
//但不允许列数不为VEC_SIZE整数倍 
template <typename T, size_t Byte>
__device__ __forceinline__ void cooperative_load_2d(
    T* __restrict__ smem_dst,           // dst, in shared memory
    const T* __restrict__ gmem_src,     // src, in global memory
    size_t ROW,                         // src数据块总行数
    size_t COL,                         // src数据块总列数; 即gmem_src shape = [ROW, COL]
    size_t row_offset,                  // 待加载数据块在src数据块中的行偏移
    size_t col_offset,                  // 待加载数据块在src数据块中的列偏移
    size_t row,                         // 实际需要加载的行数
    size_t col,                         // 实际需要加载的列数; 即待加载数据块shape = [row, col]
    size_t padding = 0                  // 实际加载时列方向的padding; 即smem_dst shape = [row, col + padding]
) {
    static_assert(Byte == 4 || Byte == 8 || Byte == 16, "Unsupported cp.async size");
    static_assert(Byte % sizeof(T) == 0, "Invalid cooperative load config");

    constexpr size_t VEC_SIZE = Byte / sizeof(T);
    int tid = threadIdx.x;
    size_t vec_per_row = col / VEC_SIZE;
    size_t vec_num = row * vec_per_row;

    // Stage 1: CTA cooperatively loads full or partial vectors.
    for (size_t i = tid; i < vec_num; i += blockDim.x) {
        size_t r = i / vec_per_row;
        size_t c = (i % vec_per_row) * VEC_SIZE;

        size_t bias_smem = r * (col + padding) + c;
        size_t global_row = row_offset + r;
        size_t global_col = col_offset + c;

        int src_size = 0;
        if (global_row < ROW && global_col < COL) {
            size_t valid_elements = min(VEC_SIZE, min(col - c, COL - global_col));
            src_size = static_cast<int>(valid_elements * sizeof(T));
        }

        // Use a valid dummy address when the entire vector is out of bounds.
        const T* src = gmem_src;
        if (src_size > 0) src = gmem_src + global_row * COL + global_col;

        cp_async<Byte>(smem_dst + bias_smem, src, src_size);
    }
}

