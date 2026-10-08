#pragma once

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