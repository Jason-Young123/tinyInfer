import os
import torch
from torch import nn

from tinyinfer.utils.context import get_context

from tinyinfer.layers.flash_attention import flash_attention # self-implemented flash attention
from tinyinfer.layers.flash_decoding import flash_decoding   # self-implemented flash decoding

# 一些独立的辅助函数
def store_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    # cache_k/v : [num_blocks, block_size, num_heads, head_dim]
    # k/v: [num_tokens, num_kv_heads, head_dim], 这里的k/v可能来自很多个seq请求且每个请求长度不一;
    # 比如seq0刚完成prefill需要存4个token对应的kv cache, 而seq1/2则都是在decode阶段、只需存储1个token;
    # 则此时num_tokens = 4 + 1 + 1 = 6
    if k.shape != v.shape:
        raise ValueError("k/v shape mismatch")
    if k.shape[0] != slot_mapping.numel():
        raise ValueError("slot_mapping length must equal token count")
    for i, slot in enumerate(slot_mapping.tolist()):
        block = slot // block_size
        offset = slot % block_size
        cache_k[block, offset].copy_(k[i])
        cache_v[block, offset].copy_(v[i])


# 针对某个请求seq获取其所有context对应的K/V cache
# cache shape: [num_blocks, block_size, num_heads, head_dim]
def gather_sequence_kv(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    if context_len <= 0:
        raise ValueError("context_len must be positive")

    chunks = [] # shape: list of [num_tokens, num_heads, head_dim]
    remaining = context_len
    
    for block_id in block_table.tolist():
        if block_id < 0 or remaining <= 0: # block_id < 0代表实际未占用block
            break
        take = min(block_size, remaining) # 最后一个block不一定恰好占满, 因此用min(block_size, remaining)避免多取
        chunks.append(cache[block_id, :take])
        remaining -= take
    if remaining != 0:
        raise RuntimeError("block_table does not cover context_len")

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

        self.num_layers = int(num_layers)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
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
        # self.storage = torch.empty(shape, dtype=dtype, device=device)
        # storage 不是模型参数，因此 register_buffer, 此后可以使用.to/.cuda()等torch的module管理方法; persistent=False 防止写入 state_dict从而不必要地保存到checkpoint里面
        self.register_buffer(
            "storage",
            torch.empty(shape, dtype=dtype, device=device),
            persistent=False,
        )

    # 获取某一层的全部KV cache
    def layer_kv(self, layer_idx: int): # return shape: [num_blocks, block_size, num_heads, head_dim]
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("layer_idx out of range")
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]




"""
attention backend选择链:

forward
  |
  +-- store_kv
  |
  +-- backend == flash
  |      |
  |      +-- _flash_attend_uniform
  |             |
  |             +-- q_len == 1
  |             |      -> flash_decoding
  |             |      -> 直接访问paged KV
  |             |
  |             +-- q_len > 1
  |                    -> gather_sequence_kv
  |                    -> flash_attention
  |
  +-- backend == torch
         |
         +-- _torch_attend_uniform
                -> gather_sequence_kv
                -> SDPA
"""




_FLASH_LOGGED = False
_TORCH_LOGGED = False

class Attention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        kv_cache: PagedKVCache | None,
        block_size: int,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.layer_idx = layer_idx
        self.num_heads = num_heads          # num of q heads
        self.num_kv_heads = num_kv_heads    # num of kv heads(可以被q整除,例如Grouped Query Attention, n个q heads对应一个kv head)
        self.head_dim = head_dim
        self.scale = scale
        self.kv_cache = kv_cache            # PagedKVCache
        self.block_size = block_size

        # 增加attention backend选择(torch or flashattention); runtime backend，不污染模型 config。
        self.backend = os.getenv("TINYINFER_ATTENTION_BACKEND", "torch").strip().lower()
        if self.backend not in {"torch", "flash"}:
            raise ValueError(
                "TINYINFER_ATTENTION_BACKEND must be 'torch' or 'flash'"
            )

    def set_kv_cache(self, kv_cache: PagedKVCache) -> None:
        if kv_cache.block_size != self.block_size:
            raise ValueError("KV cache block_size mismatch")
        self.kv_cache = kv_cache

    def _repeat(self, k:torch.Tensor, v:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_repeated = k.repeat_interleave(repeat, dim=1)
            v_repeated = v.repeat_interleave(repeat, dim=1)
            return k_repeated, v_repeated
        else:
            return k, v

    # backend = torch同一入口: 内部全部调用SPDA实现
    # unified attention computation
    def _torch_attend_uniform(
        self,
        q_i: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
        query_start_pos: int,
    ):
        # q_i:    [Tq, Hq, D] = [q_len, num_q_heads, hidden_dim]
        # cache_k: [num_blocks, block_size, num_kv_heads, hidden_dim]
        # cache_v: [num_blocks, block_size, num_kv_heads, hidden_dim]

        global _TORCH_LOGGED
        if not _TORCH_LOGGED: # 打印backend
            print("[tinyInfer] Attention backend: torch.scaled_dot_product_attention")
            _TORCH_LOGGED = True

        # 在torch backend内部materialize当前sequence对应的连续历史K/V
        k_hist = gather_sequence_kv(
            cache_k,
            block_table,
            context_len,
            self.block_size,
        ) # [Tk, Hkv, D] = [kv_len/kv_cache_len, num_kv_heads, hidden_dim]

        v_hist = gather_sequence_kv(
            cache_v,
            block_table,
            context_len,
            self.block_size,
        ) # [Tk, Hkv, D] = [kv_len/kv_cache_len, num_kv_heads, hidden_dim]

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



    # backend = flash条件下 q_len > 1专用
    def _flash_attend_prefill(self, q_i, k_hist, v_hist, query_start_pos: int):
        # q_i:    [Tq, Hq, D]
        # k_hist: [Tk, Hkv, D]
        return flash_attention(
            q_i,
            k_hist,
            v_hist,
            query_start_pos=query_start_pos,
            scale=self.scale,
            is_causal=True,
        )



    # backend = flash条件下 q_len=1专用，不materialize历史连续K/V
    def _flash_attend_decode_one(
        self,
        q_i: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        if q_i.shape[0] != 1:
            raise ValueError("_attend_decode_one requires q_len == 1")
        return flash_decoding(
            q=q_i,
            k_cache=cache_k,
            v_cache=cache_v,
            block_table=block_table,
            context_len=context_len,
            kvcache_block_size=self.block_size,
            scale=self.scale,
        )


    # backend = flash统一入口:
    # q_len > 1 -> FlashAttention
    # q_len = 1 -> Paged Flash-Decoding
    def _flash_attend_uniform(
        self,
        q_i: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
        query_start_pos: int,
    ) -> torch.Tensor:
        global _FLASH_LOGGED
        if not _FLASH_LOGGED: # 打印backend
            print("[tinyInfer] Attention backend: self implemented flash attention/decoding")
            _FLASH_LOGGED = True

        # q_i: [Tq, Hq, D]
        q_len = q_i.shape[0]

        # decode: q_len = 1，直接读取paged KV cache，不materialize连续K/V
        if q_len == 1:
            return flash_decoding(
                q=q_i,
                k_cache=cache_k,
                v_cache=cache_v,
                block_table=block_table,
                context_len=context_len,
                kvcache_block_size=self.block_size,
                scale=self.scale,
            )

        # prefill/chunked prefill: q_len > 1，暂时仍materialize连续历史K/V
        k_hist = gather_sequence_kv(
            cache_k,
            block_table,
            context_len,
            self.block_size,
        )

        v_hist = gather_sequence_kv(
            cache_v,
            block_table,
            context_len,
            self.block_size,
        )

        return flash_attention(
            q_i,
            k_hist,
            v_hist,
            query_start_pos=query_start_pos,
            scale=self.scale,
            is_causal=True,
        )




    # interface
    def forward(self, q, k, v): # q/k/v shape: [num_tokens, num_heads/num_kv_heads, head_dim], 这里代表刚刚乘过Wq/k/v的向量
        ctx = get_context()
        cache_k, cache_v = self.kv_cache.layer_kv(self.layer_idx) # 获取所有KV cache slot

        # 先把本轮所有新 K/V 写入 paged cache; [CRITICAL]
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

        for i in range(len(cu_q) - 1): # 对这一轮所有需要处理的seq请求逐一调用attention计算
            qs, qe = cu_q[i], cu_q[i + 1]
            q_i = q[qs:qe]

            context_len = int(ctx.context_lens[i].item())
            q_len = qe - qs
            query_start = context_len - q_len

            # flash backend:
            # q_len > 1 -> gather连续K/V后进入FlashAttention
            # q_len = 1 -> 直接进入Paged Flash-Decoding, 不materialize连续K/V
            if self.backend == "flash":
                out_i = self._flash_attend_uniform(
                    q_i=q_i,
                    cache_k=cache_k,
                    cache_v=cache_v,
                    block_table=ctx.block_tables[i],
                    context_len=context_len,
                    query_start_pos=query_start,
                )

            # torch backend:
            # 始终在backend内部gather连续K/V后进入SDPA
            elif self.backend == "torch":
                out_i = self._torch_attend_uniform(
                    q_i=q_i,
                    cache_k=cache_k,
                    cache_v=cache_v,
                    block_table=ctx.block_tables[i],
                    context_len=context_len,
                    query_start_pos=query_start,
                )

            else:
                raise ValueError(f"unsupported attention backend: {self.backend}")

            outputs.append(out_i)

        return torch.cat(outputs, dim=0) # shape = [num_total_tokens, num_heads, head_dim]
