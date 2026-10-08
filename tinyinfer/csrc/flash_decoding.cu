#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>


// 参考flash decoding, 同时引入paged attention的block寻址
// 并行策略: 
//   - grid_dim 采用二维 = [num_head, ceil(num_kvcache_block)], 即每个block处理一个head其中的一段完整的KV block @ q, 这样更利于coalesced memory access
//   - block_dim 固定为128 (即4个warp), 每个warp轮流处理一个token (考虑head_dim通常为32的整数倍), 直到遍历完成kvcache_block_size个token
//		例如, 当kvcache_block_size = 16时, warp0负责token0/4/8/12, warp1负责token1/5/9/13, ...
// 类似flashAttention, softmax进行在线更新
template <typename T> 
__global__ void kernel_flash_decoding(
	int src_seq_len, // 本轮所需要的全部历史kv cache长度; src_seq_len + 1 = tgt_seq_len(仅考虑单个token的decode)
	int q_heads,
	int kv_heads,
	int head_dim,
	float scale,

	const T* q, // 注意尺寸为[1, q_heads, head_dim], 仅对应一个token
	const T* KCache, // 完整的 K Cache; 尺寸为 [num_blocks, block_size, kv_heads, head_dim]
	const T* VCache, // 完整的 V Cache; 尺寸为 [num_blocks, block_size, kv_heads, head_dim]
	const int* table_list, // logical -> physical block mapping
	const int num_kvcache_block, // 等于 len(table_list)
	const int kvcache_block_size, // 一个kvcache block包含多少个token

	T* l,
	T* m, 
	T* O
) {



}




