#include "utils.h"

// Flash-Decoding 固定一个 CTA = 4 warps = 128 threads。
constexpr int FLASH_DECODING_NUM_WARPS = 4;
constexpr int FLASH_DECODING_THREADS = 128;
constexpr unsigned FLASH_DECODING_MASK = 0xffffffffu;



// 一个 warp 协作完成一个 token 的 q·k。
__device__ __forceinline__ float warp_reduce_sum_decoding(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FLASH_DECODING_MASK, val, offset);
    }
    return __shfl_sync(FLASH_DECODING_MASK, val, 0);
}





// 参考flash decoding, 同时引入paged attention的block寻址
// 并行策略: 
//   - grid_dim 采用二维 = [num_head, ceil(num_kvcache_block)], 即每个block处理一个head其中的一段完整的KV block @ q, 这样更利于coalesced memory access
//   - block_dim 固定为128 (即4个warp), 每个warp轮流处理一个token (考虑head_dim通常为32的整数倍), 直到遍历完成kvcache_block_size个token
//		例如, 当kvcache_block_size = 16时, warp0负责token0/4/8/12, warp1负责token1/5/9/13, ...
// 类似flashAttention, softmax进行在线更新

// Stage 1: 每个block处理一个head的一个kvcache_block
template <typename T> 
__global__ void kernel_flash_decoding(
	int context_len, // 本轮所需要的全部历史kv cache长度
	int q_heads,
	int kv_heads,
	int head_dim,
	float scale,

	const T* __restrict__ q, // 注意尺寸为[1, q_heads, head_dim], 仅对应一个token
	const T* __restrict__ KCache, // 完整的 K Cache; 尺寸为 [num_blocks, block_size, kv_heads, head_dim]
	const T* __restrict__ VCache, // 完整的 V Cache; 尺寸为 [num_blocks, block_size, kv_heads, head_dim]
	const int* __restrict__ table_list, // logical -> physical block mapping
	int num_kvcache_block, // 等于 len(table_list)
	int kvcache_block_size, // 一个kvcache block包含多少个token

	// 传给第二个kernel进行fuse; 为了确保精度全部用float保护
	float* __restrict__ partial_l, // [q_heads, num_kvcache_block]
    float* __restrict__ partial_m, // [q_heads, num_kvcache_block]
    float* __restrict__ partial_O //  [q_heads, num_kvcache_block, head_dim]
) {
	// grid_dim = [num_q_heads, num_kvcache_blocks]
	// 每个block负责一个head的一个kvcache_block, 即 [1, head_dim] @ [head_dim, kvcache_block_size]的矩阵乘法
	const int q_head = blockIdx.x;
    const int logical_block = blockIdx.y;
    const int tid = threadIdx.x; 
    const int warp_id = tid >> 5; // 一个warp负责一个token
    const int lane_id = tid & 31;

	// basic checks
	if (q_head >= q_heads || logical_block >= num_kvcache_block) {
        return;
    }
    if (blockDim.x != FLASH_DECODING_THREADS) { // 固定 block.x=128，host launcher 同样固定该值
        return;
    }

    // GQA 映射: 多个 Q heads 共享一个 KV head
    const int q_per_kv = q_heads / kv_heads;
    const int kv_head = q_head / q_per_kv;

    // 当前 logical KV block 对应的全局 token 起点
    const int token_start = logical_block * kvcache_block_size;
    if (token_start >= context_len) {
        return;
    }
    const int valid_tokens = min(kvcache_block_size, context_len - token_start); // tail process

    // PagedAttention 核心: logical -> physical block
    const int physical_block = table_list[logical_block]; // 后续从K/V_cache[physical_block]中fetch hist-K/V
    if (physical_block < 0) {
        return;
    }


	// shared memory layout:
    // q_sm[head_dim]
    // warp_O[4][head_dim]
    // warp_m[4]
    // warp_l[4]
    // block_m[1]
    // block_l[1]
    extern __shared__ float shared_mem[];
    float* q_sm = shared_mem;
    float* warp_O = q_sm + head_dim;
    float* warp_m = warp_O + FLASH_DECODING_NUM_WARPS * head_dim;
    float* warp_l = warp_m + FLASH_DECODING_NUM_WARPS;
    float* block_m = warp_l + FLASH_DECODING_NUM_WARPS;
    float* block_l = block_m + 1;



	// Step 1: Initialization
	// Q 只有一个 token，整个 CTA 协作缓存当前 q_head @ [1, head_dim]
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_sm[d] = to_float<T>(q[q_head * head_dim + d]);
    }
    // 每个 warp 独立维护一个 partial output state
    for (int d = lane_id; d < head_dim; d += 32) { // 每个warp初始化warp_O的一行
        warp_O[warp_id * head_dim + d] = 0.0f;
    }
    if (lane_id == 0) { // 每个warp初始化warp_m/warp_l的一个元素
        warp_m[warp_id] = -INFINITY;
        warp_l[warp_id] = 0.0f;
    }
    __syncthreads();
    
	float local_m = -INFINITY; // 设置每个warp的初始化local_m/local_l
    float local_l = 0.0f;

    // Step 2: 4 个 warp 以步长 4 轮流处理 token
	// 这里已经是在线更新/第一层合并, warp-strided对多个token进行合并
    for (int token_off = warp_id; token_off < valid_tokens; token_off += FLASH_DECODING_NUM_WARPS) {
        // K/V layout:
        // [physical_block, token_off, kv_head, head_dim]
        const size_t kv_base =
            (
                (
                    static_cast<size_t>(physical_block)
                    * static_cast<size_t>(kvcache_block_size)
                    + static_cast<size_t>(token_off)
                )
                * static_cast<size_t>(kv_heads)
                + static_cast<size_t>(kv_head)
            )
            * static_cast<size_t>(head_dim);

        // Step 2.1: 一个 warp 协作完成 q·k
        float dot = 0.0f;
        for (int d = lane_id; d < head_dim; d += 32) {
            const float q_val = q_sm[d];
            const float k_val = to_float<T>(KCache[kv_base + static_cast<size_t>(d)]);
            dot += q_val * k_val;
        }
        dot = warp_reduce_sum_decoding(dot);
        const float score = dot * scale;

        // Step 2.2: warp-local online softmax。
        const float new_m = fmaxf(local_m, score);
        const float alpha = (local_l == 0.0f) ? 0.0f : myexp(local_m - new_m);
        const float beta = myexp(score - new_m);

        // Step 2.3: O <- alpha * O + beta * V; 迭代更新O
        for (int d = lane_id; d < head_dim; d += 32) {
            const float v_val = to_float<T>(VCache[kv_base + static_cast<size_t>(d)]);
            float& out = warp_O[warp_id * head_dim + d];
            out = alpha * out + beta * v_val; // out是引用, 因此实际上更新了warp_O中的内容
        }

		// Step 2.4: 迭代更新local_l和local_m
        local_l = alpha * local_l + beta;
        local_m = new_m;
    }

     // 由thread0整理每个warp最终的m和l; 理论上4个warp的thread0都会写warp_m, 但实际没参与的warp会往其中写入初始值-∞/0
    if (lane_id == 0) {
        warp_m[warp_id] = local_m;
        warp_l[warp_id] = local_l;
    }
    __syncthreads();

	// 现在开始第二层合并, 固定把4个warp的m/l/O合并为最终一个
    // Step 3: CTA 内先合并 4 个 warp 的 m/l
    if (tid == 0) {
        float merged_m = -INFINITY;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) { // 滤除未参与的warp
                merged_m = fmaxf(merged_m, warp_m[w]);
            }
        }

        float merged_l = 0.0f;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) { // 滤除未参与的warp
                merged_l += myexp(warp_m[w] - merged_m) * warp_l[w];
            }
        }
        block_m[0] = merged_m;
        block_l[0] = merged_l;
    }
    __syncthreads();

    const float merged_m = block_m[0]; //每个thread都获取到合并后的m和l
    const float merged_l = block_l[0];


    // Step 4: CTA 内合并 4 个 warp 的 numerator O; 由一个block负责
	// partial_idx为[q_head, logical_block]展平后的结果
	const size_t partial_idx = static_cast<size_t>(q_head) * static_cast<size_t>(num_kvcache_block) + static_cast<size_t>(logical_block);
    
	for (int d = tid; d < head_dim; d += blockDim.x) {
        float merged_O = 0.0f;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                const float correction = myexp(warp_m[w] - merged_m);
                merged_O += correction * warp_O[w * head_dim + d];
            }
        }
        partial_O[partial_idx * static_cast<size_t>(head_dim) + static_cast<size_t>(d)] = merged_O;
    }

    if (tid == 0) {
        partial_m[partial_idx] = merged_m; // [q_heads, num_kvcache_block]
        partial_l[partial_idx] = merged_l; // [q_heads, num_kvcache_block]
    }

}



// Stage 2: 一个 CTA 负责一个 q_head，合并所有 KV-block partial states
// 整体流程和Stage1完全相同, 分为两层合并
template <typename T>
__global__ void kernel_flash_decoding_reduce(
    int q_heads,
    int head_dim,
    int num_kvcache_block,

    const float* __restrict__ partial_l, // [q_heads, num_kvcache_blocks]
    const float* __restrict__ partial_m, // [q_heads, num_kvcache_blocks]
    const float* __restrict__ partial_O, // [q_heads, num_kvcache_blocks, head_dim]

    T* __restrict__ O
) {
    const int q_head = blockIdx.x; // 每个block负责一个head; 每个block固定启用128个线程

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (q_head >= q_heads) {
        return;
    }
    if (blockDim.x != FLASH_DECODING_THREADS) {
        return;
    }

    // shared memory layout:
    // warp_O[4][head_dim]
    // warp_m[4]
    // warp_l[4]
    // final_m[1]
    // final_l[1]
    extern __shared__ float shared_mem[];
    float* warp_O = shared_mem;
    float* warp_m = warp_O + FLASH_DECODING_NUM_WARPS * head_dim;
    float* warp_l = warp_m + FLASH_DECODING_NUM_WARPS;
    float* final_m = warp_l + FLASH_DECODING_NUM_WARPS;
    float* final_l = final_m + 1;

	// Initialization
    for (int d = lane_id; d < head_dim; d += 32) {
        warp_O[warp_id * head_dim + d] = 0.0f;
    }
    if (lane_id == 0) {
        warp_m[warp_id] = -INFINITY;
        warp_l[warp_id] = 0.0f;
    }
    __syncthreads();

    float local_m = -INFINITY;
    float local_l = 0.0f;

    // 4 个 warp 同样以步长 4 轮流处理 Stage-1 partial blocks。
    // 第一层合并
	for (int block_id = warp_id; block_id < num_kvcache_block; block_id += FLASH_DECODING_NUM_WARPS) {
        const size_t partial_idx =
            static_cast<size_t>(q_head)
            * static_cast<size_t>(num_kvcache_block)
            + static_cast<size_t>(block_id);

        const float block_l_value = partial_l[partial_idx];

        if (block_l_value <= 0.0f) {
            continue;
        }

        const float block_m_value = partial_m[partial_idx];
        const float new_m = fmaxf(local_m, block_m_value);
        const float alpha = (local_l == 0.0f) ? 0.0f : myexp(local_m - new_m);
        const float beta = myexp(block_m_value - new_m);

        // partial_O 本身是以 block_m 为基准的 numerator，不能提前 / block_l
        for (int d = lane_id; d < head_dim; d += 32) {
            const float block_O = partial_O[partial_idx * static_cast<size_t>(head_dim) + static_cast<size_t>(d)];
            float& out = warp_O[warp_id * head_dim + d];
            out = alpha * out + beta * block_O;
        }
        local_l = alpha * local_l + beta * block_l_value;
        local_m = new_m;
    }

    if (lane_id == 0) {
        warp_m[warp_id] = local_m;
        warp_l[warp_id] = local_l;
    }

    __syncthreads();

    // 先合并4个warp的全局 m/l
	// 第二层合并
    if (tid == 0) {
        float merged_m = -INFINITY;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_m = fmaxf(merged_m, warp_m[w]);
            }
        }

        float merged_l = 0.0f;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                merged_l += myexp(warp_m[w] - merged_m) * warp_l[w];
            }
        }
        final_m[0] = merged_m;
        final_l[0] = merged_l;
    }
    __syncthreads();

    const float merged_m = final_m[0];
    const float merged_l = final_l[0];

    // 最后合并 numerator，并且只在这里执行 O /= l
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float merged_O = 0.0f;
		#pragma unroll
        for (int w = 0; w < FLASH_DECODING_NUM_WARPS; ++w) {
            if (warp_l[w] > 0.0f) {
                const float correction = myexp(warp_m[w] - merged_m);
                merged_O += correction * warp_O[w * head_dim + d];
            }
        }
        O[q_head * head_dim + d] = from_float<T>(merged_O / merged_l);
    }
}






// 对上述两个kernel进行发射
// dtype-specialized launcher
template <typename T>
void launch_flash_decoding_typed(
    const T* q,
    const T* k_cache,
    const T* v_cache,
    const int* block_table,
    T* out,
    float* partial_l, //注意partial_l/m/O位于GM, 是连接stage1和stage2的中间桥梁
    float* partial_m,
    float* partial_O,
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int num_kvcache_block,
    int kvcache_block_size,
    float scale
) {
	// launch stage1: 
    const dim3 stage1_grid(q_heads, num_kvcache_block);
    const dim3 stage1_block(FLASH_DECODING_THREADS);
    const size_t stage1_smem_size =
        (	head_dim
            + FLASH_DECODING_NUM_WARPS * head_dim
            + FLASH_DECODING_NUM_WARPS
            + FLASH_DECODING_NUM_WARPS
            + 2
        ) * sizeof(float);

    kernel_flash_decoding<T><<<stage1_grid, stage1_block, stage1_smem_size>>>(
        context_len,
        q_heads,
        kv_heads,
        head_dim,
        scale,
        q,
        k_cache,
        v_cache,
        block_table,
        num_kvcache_block,
        kvcache_block_size,
        partial_l,
        partial_m,
        partial_O
    );
    CUDA_CHECK(cudaGetLastError());


	// launch stage2:
    const dim3 stage2_grid(q_heads);
    const dim3 stage2_block(FLASH_DECODING_THREADS);
    const size_t stage2_smem_size =
        (	FLASH_DECODING_NUM_WARPS * head_dim
            + FLASH_DECODING_NUM_WARPS
            + FLASH_DECODING_NUM_WARPS
            + 2
        ) * sizeof(float);

    kernel_flash_decoding_reduce<T><<<stage2_grid, stage2_block, stage2_smem_size>>>(
        q_heads,
        head_dim,
        num_kvcache_block,
        partial_l,
        partial_m,
        partial_O,
        out
    );
    CUDA_CHECK(cudaGetLastError());
}



// 对外接口
// C++ binding 只需要传裸 device pointer 与 runtime metadata。
extern "C" void flash_decoding_forward_cuda(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int* block_table,
    void* out,
    float* partial_l,
    float* partial_m,
    float* partial_O,
    int dtype_code,
    int context_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int num_kvcache_block,
    int kvcache_block_size,
    float scale
) {
    if (dtype_code == 0) {
        launch_flash_decoding_typed<half>(
            static_cast<const half*>(q),
            static_cast<const half*>(k_cache),
            static_cast<const half*>(v_cache),
            block_table,
            static_cast<half*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }

    if (dtype_code == 1) {
        launch_flash_decoding_typed<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k_cache),
            static_cast<const __nv_bfloat16*>(v_cache),
            block_table,
            static_cast<__nv_bfloat16*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }

    if (dtype_code == 2) {
        launch_flash_decoding_typed<float>(
            static_cast<const float*>(q),
            static_cast<const float*>(k_cache),
            static_cast<const float*>(v_cache),
            block_table,
            static_cast<float*>(out),
            partial_l,
            partial_m,
            partial_O,
            context_len,
            q_heads,
            kv_heads,
            head_dim,
            num_kvcache_block,
            kvcache_block_size,
            scale
        );
        return;
    }
    throw std::runtime_error("unsupported dtype_code");
}



