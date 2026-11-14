#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
__device__ __forceinline__ float decode_e8m0(uint8_t code) {
    if (code == 255) return __int_as_float(0x7fc00000);
    return ldexpf(1.0f, int(code) - 127);
}


// fp4 e2m1 -> fp32
__device__ __forceinline__ float decode_e2m1(uint32_t code) {
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
