// 和 mma (SM -> Reg) 指令相关函数

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>


// 有关ldmatrix与mma.sync
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 4个8x8 BF16矩阵，构造A Fragment
__device__ __forceinline__ void ldmatrix_a4(uint32_t (&a)[4], const __nv_bfloat16* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem_u32(p)) : "memory");
}

// 2个8x8 BF16矩阵，构造B Fragment；当前[N,K]布局无需trans
__device__ __forceinline__ void ldmatrix_b2(uint32_t (&b)[2], const __nv_bfloat16* p) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(smem_u32(p)) : "memory");
}

// C[16,8] += A[16,16] * B[16,8]，BF16输入、FP32累加
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}