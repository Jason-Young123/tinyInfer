#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/std/numeric>

#include <cmath>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h> //ensure csrc to use the same stream

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


__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}


// fp4-quant related; scale: e8m0 -> fp32
/*__device__ __forceinline__ float decode_e8m0(uint8_t code) {
    if (code == 255) return __int_as_float(0x7fc00000);
    return ldexpf(1.0f, int(code) - 127);
}*/

__device__ __forceinline__ float decode_e8m0(uint8_t code) {
    uint32_t special = (code == 0u) | (code == 255u);
    uint32_t bits = (static_cast<uint32_t>(code) << 23) | (special << 22);

    return __uint_as_float(bits);
}


// fp4 e2m1 -> fp32
/*__device__ __forceinline__ float decode_e2m1(uint32_t code) {
    uint32_t mag = code & 0x7;
    uint32_t sign = code & 0x8;

    float val;
    switch (mag) {
        case 0: val = 0.0f; break;
        case 1: val = 0.5f; break;
        case 2: val = 1.0f; break;
        case 3: val = 1.5f; break;
        case 4: val = 2.0f; break;
        case 5: val = 3.0f; break;
        case 6: val = 4.0f; break;
        default: val = 6.0f; break;
    }

    return sign ? -val : val;
}*/

__device__ __forceinline__ float decode_e2m1(uint32_t code) {
    uint32_t mag = code & 0x7u;

    uint32_t exponent = 126u + (mag >> 1);
    uint32_t mantissa = (mag & 1u) << 22;

    exponent &= 0u - static_cast<uint32_t>(mag != 0u);
    mantissa &= 0u - static_cast<uint32_t>(mag >= 2u);

    uint32_t bits = (exponent << 23) | mantissa | ((code & 0x8u) << 28);

    return __uint_as_float(bits);
}



template <typename T>
__device__ inline T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        //val += __shfl_down_sync(0xffffffff, val, offset);
        val += __shfl_xor_sync(0xffffffff, val, offset);
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














// 有关smem访问;
// 注意constexpr自带inline, 可以直接放在.h文件中;
// __host__代表该函数可以由cpu端函数直接调用; __device__代表该函数可以由__global__或者其他__device__函数调用;
// __device__本身不代表该函数一定在gpu端运行, 也可以在编译期由cpu调用
// 常见smem访存场景下寻找最小的 bank-conflict-free stride bytes
__host__ __device__ constexpr int get_min_stride_bytes(
    int row_bytes,              // 原本每一行有多少Bytes
    int lanes_per_row,          // 多少个lane负责取同一行的数据
    int bytes_per_lane,         // 每个lane一次取多少bytes
    int alignment_bytes = 16    // 最终的stride需要满足多少字节对齐
) {
    // 仅支持 lanes_per_row = 1/2/4/8/16/32
    if (lanes_per_row <= 0 || 32 % lanes_per_row != 0) return -1;
    // 仅支持 bytes_per_lane = 4/8/16
    if(!(bytes_per_lane == 4 || bytes_per_lane == 8 || bytes_per_lane == 16)) return -1;
    if (alignment_bytes <= 0 || row_bytes <= 0) return -1;

    int V = bytes_per_lane;
    int C = lanes_per_row * V;
    int A = alignment_bytes;

    if (C > row_bytes) return -1;

    // C >= 128, 直接向上对齐至lcm(A, V) byte即可
    if (C >= 128) {
        int X = cuda::std::lcm(A, V);
        return (row_bytes + X - 1) / X * X;
    }

    //计算stride逻辑; stride需要同时为C的奇数倍和A的整数倍, 且不小于row_bytes
    int X = cuda::std::lcm(A, C);
    if ((X / C) % 2 == 0) {
        return -1;
    }
    else {
        int q = (row_bytes + X - 1) / X;
        return (q | 1) * X;
    }
}






