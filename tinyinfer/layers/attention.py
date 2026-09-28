import torch
from torch import nn

from tinyinfer.utils.context import get_context

# 一些独立的辅助函数
def store_kv(cache_k, cache_v, k, v, slot_mapping, block_size):
    # cache_k/v : [num_blocks, block_size, num_heads, head_dim]
    # k/v: [num_tokens, num_kv_heads, head_dim], 这里的k/v可能来自很多个seq请求且每个请求长度不一;
    # 比如seq0刚完成prefill需要存4个token对应的kv cache, 而seq1/2则都是在decode阶段、只需存储1个token;
    # 则此时num_tokens = 4 + 1 + 1 = 6
    for i, slot in enumerate(slot_mapping.tolist()):
        block = slot // block_size
        offset = slot % block_size
        cache_k[block, offset].copy_(k[i])
        cache_v[block, offset].copy_(v[i])


# 针对某个请求seq获取其所有context对应的K/V cache
# cache shape: [num_blocks, block_size, num_heads, head_dim]
def gather_sequence_kv(cache, block_table, context_len, block_size):
    chunks = [] # shape: list of [num_tokens, num_heads, head_dim]
    remaining = context_len
    for block_id in block_table.tolist():
        if block_id < 0 or remaining <= 0: # block_id < 0代表实际未占用block
            break
        take = min(block_size, remaining) # 最后一个block不一定恰好占满, 因此用min(block_size, remaining)避免多取
        chunks.append(cache[block_id, :take])
        remaining -= take
    return torch.cat(chunks, dim=0) # shape: [total_num_tokens/context_len, num_heads, head_dim]



class PagedKVCache(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        shape = (
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        # 这里是完整的KV cache存储空间, = #layers * #heads * head_dim * max_seq_len * sizeof(float16) * 2
        # 其中 max_seq_len = #blocks * block_size
        # storage shape: [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
        self.storage = torch.empty(shape, dtype=dtype, device=device)

    # 获取某一层的全部KV cache
    def layer_kv(self, layer_idx: int): # return shape: [num_blocks, block_size, num_heads, head_dim]
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]



class Attention(nn.Module):
    def __init__(self, layer_idx, num_heads, num_kv_heads, head_dim, scale, kv_cache, block_size):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads          # num of q heads
        self.num_kv_heads = num_kv_heads    # num of kv heads(可以被q整除,例如Grouped Query Attention, n个q heads对应一个kv head)
        self.head_dim = head_dim
        self.scale = scale
        self.kv_cache = kv_cache            # PagedKVCache
        self.block_size = block_size

    def _repeat(self, k, v):
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_repeated = k.repeat_interleave(repeat, dim=1)
            v_repeated = v.repeat_interleave(repeat, dim=1)
            return k_repeated, v_repeated
        else:
            return k, v

    # 目前教学版的prefill还没有考虑prefix cache的拼接问题
    # q/k/v shape: [num_tokens, num_heads, head_dim]
    def _prefill(self, q, k, v, ctx):
        outputs = []
        cu_q = ctx.cu_seqlens_q.tolist()

        for i in range(len(cu_q) - 1):
            qs, qe = cu_q[i], cu_q[i + 1]
            qi = q[qs:qe]  # [Tq, Hq, D] = [seq_len, num_heads, head_dim]
            ki = k[qs:qe]
            vi = v[qs:qe]

            # [1, H, T, D] = [1, num_heads, seq_len, head_dim]
            qi = qi.transpose(0, 1).unsqueeze(0)
            ki = ki.transpose(0, 1).unsqueeze(0)
            vi = vi.transpose(0, 1).unsqueeze(0)

            # 教学版只处理无 cached-prefix 的最基本情况；
            # cached-prefix + new suffix 在下一小节扩展。
            ki, vi = self._repeat(ki, vi)
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True, scale=self.scale
            )
            outputs.append(oi.squeeze(0).transpose(0, 1)) # [1, H, T, D] -> [T, H, D] = [seq_len, num_heads, head_dim]

        return torch.cat(outputs, dim=0) # [total_seq_len, num_heads, head_dim]

    # 教学版本decode
    # q shape: [num_tokens, num_heads, head_dim], 注意这里num_tokens = 本batch seq请求数目, 因为通常默认decode一次只推理一个token
    def _decode(self, q, cache_k, cache_v, ctx):
        outputs = []
        for i in range(q.shape[0]): # 对于其中的第i个seq请求
            k_hist = gather_sequence_kv( # [context_len, num_heads, head_dim]
                cache_k, # [num_blocks, block_size, num_heads, head_dim]
                ctx.block_tables[i], # 该请求的完整block_table列表
                int(ctx.context_lens[i].item()), # 该请求的上下文总长
                self.block_size,
            )
            v_hist = gather_sequence_kv(
                cache_v,
                ctx.block_tables[i],
                int(ctx.context_lens[i].item()),
                self.block_size,
            )
            qi = q[i:i+1].transpose(0, 1).unsqueeze(0) # 一个seq对应一个q token, shape = [1, num_heads, 1, head_dim]; 注意[i:i+1]不会消去维度,和[i]不同
            ki = k_hist.transpose(0, 1).unsqueeze(0) # [1, num_heads, context_len, head_dim]
            vi = v_hist.transpose(0, 1).unsqueeze(0) # [1, num_heads, context_len, head_dim]

            ki, vi = self._repeat(ki, vi)
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, is_causal=False, scale=self.scale
            )
            outputs.append(oi.squeeze(0).transpose(0, 1)) # [1, num_heads, 1, head_dim] -> [1, num_heads, head_dim]
    
        return torch.cat(outputs, dim=0) # [total_seq_num/total_request_num, num_heads, head_dim]

    
    # q/k/v shape: [num_tokens, num_heads, head_dim]
    def forward(self, q, k, v): # 这里的q/k/v可能对应多个token, 但只考虑一层
        ctx = get_context() # 调用独立函数获取推理ctx
        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx) # 获取某一层的全部KV Cache
        store_kv(cache_k, cache_v, k, v, ctx.slot_mapping, self.block_size)

        if ctx.is_prefill:
            return self._prefill(q, k, v, ctx)
        else:
            return self._decode(q, cache_k, cache_v, ctx)


    


    # unified attention computation
    def _attend_one(self, q_i, k_hist, v_hist, query_start_pos: int):
        # q_i:    [Tq, Hq, D] = [q_len, num_q_heads, hidden_dim]
        # k_hist: [Tk, Hkv, D] = [kv_len/kv_cache_len, num_kv_heads, hidden_dim]

        q = q_i.transpose(0, 1).unsqueeze(0)    # [1, num_q_heads, q_len, hidden_dim]
        k = k_hist.transpose(0, 1).unsqueeze(0) # [1, num_kv_heads, kv_cache_len, hidden_dim]
        v = v_hist.transpose(0, 1).unsqueeze(0) # [1, num_kv_heads, kv_cache_len, hidden_dim]

        k, v = self._repeat(k, v)               # [1, num_q_heads, kv_cache_len, hidden_dim]

        tq = q_i.shape[0]
        tk = k_hist.shape[0]

        q_pos = torch.arange( # e.g. [10, 15], 其中历史kv cache长度为10, 本轮需要计算的token数为5; shape = [5]
            query_start_pos,
            query_start_pos + tq,
            device=q.device,
        )
        k_pos = torch.arange(tk, device=q.device) # e.g. 10, shape = [10]

        causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) # shape = [5, 10]
        # shape = [1, 1, 5, 10] = [Batch, heads/q_heads, q_len, kv_cache_len], 用于后续attention计算时自动广播对齐
        causal = causal.unsqueeze(0).unsqueeze(0) 

        out = torch.nn.functional.scaled_dot_product_attention( # [1, num_q_heads, q_len, hidden_dim]
            q,                  # [1, num_q_heads, q_len, hidden_dim]
            k,                  # [1, num_q_heads, kv_cache_len, hidden_dim]
            v,                  # [1, num_q_heads, kv_cache_len, hidden_dim]
            attn_mask=causal,   # 自行提供casual mask
            is_causal=False,    # 不需要torch自动计算mask
            scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1) # [q_len, num_q_heads, hidden_dim]


    def forward(self, q, k, v): # q/k/v shape: [num_tokens, num_heads, head_dim]
        ctx = get_context()
        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx) # 获取所有KV cache slot

        # 先把本轮所有新 K/V 写入 paged cache。
        store_kv(
            cache_k,
            cache_v,
            k,
            v,
            ctx.slot_mapping,
            self.block_size,
        )

        cu_q = ctx.cu_seqlens_q.tolist()
        outputs = []

        for i in range(len(cu_q) - 1): # 对这一轮所有需要处理的seq请求逐一调用_attent_one进行attention计算
            qs, qe = cu_q[i], cu_q[i + 1]
            q_i = q[qs:qe]

            context_len = int(ctx.context_lens[i].item())
            q_len = qe - qs
            query_start = context_len - q_len

            k_hist = gather_sequence_kv(
                cache_k,
                ctx.block_tables[i],
                context_len,
                self.block_size,
            )
            v_hist = gather_sequence_kv(
                cache_v,
                ctx.block_tables[i],
                context_len,
                self.block_size,
            )

            outputs.append(
                self._attend_one(
                    q_i,
                    k_hist,
                    v_hist,
                    query_start_pos=query_start,
                )
            )

        return torch.cat(outputs, dim=0)
