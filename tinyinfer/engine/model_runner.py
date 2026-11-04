from dataclasses import dataclass
import torch

from tinyinfer.utils.context import set_context, reset_context
from tinyinfer.engine.sequence import SchedulerOutput
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights
from tinyinfer.engine.speculative import speculative_verify


DEBUG = True # for debug mode only


# 将seq内的逻辑token position映射到paged KV的flat slot
def token_to_slot(seq, token_position: int) -> int:
    block_size = seq.block_size
    logical_block = token_position // block_size
    offset = token_position % block_size
    if logical_block >= len(seq.block_table):
        raise RuntimeError("token position has no allocated physical block")
    physical_block = seq.block_table[logical_block]
    return physical_block * block_size + offset


def dtype_nbytes(dtype:torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

# 计算一个token在所有层中所需要的KV cache容量大小
def bytes_per_kv_block(hf_config, block_size: int, dtype: torch.dtype) -> int:
    head_dim = int(
        getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
    )
    return (
        int(hf_config.num_hidden_layers)
        * 2  # K + V
        * int(block_size)
        * int(hf_config.num_key_value_heads)
        * head_dim
        * dtype_nbytes(dtype)
    )


# Resource Statistics 统一采用同一套预算口径，五部分严格相加等于 GPU 总显存：
# Reserved: TOTAL_MEM * (1 - gpu_memory_utilization), 完全不使用的预留区域
# Weights: 模型权重
# Temporary Peak: 一次最大token数prefill过程中temporary activation所占用的总显存(不包含KV cache)
# KV budget: 包含Allocated和Available, 分别代表KV cache已经占用和还可以占用的显存大小
# Reserved + Weights + Temporary Peak + Allocated for KV + Available for KV = TOTAL_MEM
@dataclass(frozen=True, slots=True)
class KVMemoryProfile:
    total_bytes: int
    budget_bytes: int
    reserved_bytes: int
    weights_bytes: int
    temporary_peak_bytes: int
    block_bytes: int
    num_blocks: int
    target_block_bytes: int = 0
    draft_block_bytes: int = 0
    target_weights_bytes: int = 0
    draft_weights_bytes: int = 0



# 测评单一模型(draft或者target model)的动态activation peak/temporary peak, 
# 即当输入seq_len取最大值(max_num_batched_tokens)时的动态activations占用的总显存
def _profile_one_model_peak(
    model, 
    hf_config, 
    config, 
    device: torch.device, 
    dtype: torch.dtype, 
    weights_bytes: int
) -> tuple[int, int]:
    n = config.max_num_batched_tokens
    if n <= 0:
        raise ValueError("profile token count must be positive")
    
    block_size = config.kvcache_block_size
    num_dummy_blocks = (n + block_size - 1) // block_size
    head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
    dummy_cache = PagedKVCache( # 创建相应尺寸的dummy kv cache
        num_layers=hf_config.num_hidden_layers,
        num_blocks=num_dummy_blocks,
        block_size=block_size,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    model.set_kv_cache(dummy_cache)
    dummy_bytes = dummy_cache.size()

    input_ids = torch.zeros(n, dtype=torch.long, device=device)     # 创建相应尺寸的dummy input
    positions = torch.arange(n, dtype=torch.long, device=device)
    hidden = None
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        set_context(    # 准备单已请求batch的metadata
            q_lens=torch.tensor([n], dtype=torch.int32, device=device),
            context_lens=torch.tensor([n], dtype=torch.int32, device=device),
            cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=device),
            max_seqlen_q=n,
            slot_mapping=torch.arange(n, dtype=torch.long, device=device),
            block_tables=torch.arange(num_dummy_blocks, dtype=torch.int32, device=device).unsqueeze(0),
            is_prefill=torch.tensor([True], dtype=torch.bool, device=device),
            is_sample=torch.tensor([True], dtype=torch.bool, device=device),
        )
        with torch.inference_mode():
            hidden, _ = model(input_ids, positions, output_hidden_states=False)
            model.compute_logits(hidden)
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device)) # 获取本次前向传播过程中的peak memory
        
    finally:
        reset_context()
        for layer in model.model.layers:
            layer.self_attn.attn.kv_cache = None
        del dummy_cache, input_ids, positions, hidden # 清除所有临时数据
        torch.cuda.empty_cache()
    return max(0, peak - weights_bytes - dummy_bytes), int(dummy_bytes) # [temporary peak, cache size]





# 在SD模式/标准模式下对memory进行profiling
def build_kv_memory_profile(
    config, 
    device: torch.device, 
    dtype: torch.dtype, 
    target_model, 
    draft_model, 
) -> KVMemoryProfile:
    if device.type != "cuda":
        raise ValueError("Invalid device: cpu; Required: cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    budget_bytes = int(total_bytes * config.gpu_memory_utilization)
    reserved_bytes = total_bytes - budget_bytes

    target_weights_bytes = sum(p.numel() * p.element_size() for p in target_model.parameters())
    draft_weights_bytes = sum(p.numel() * p.element_size() for p in draft_model.parameters()) if config.speculative else 0
    weights_bytes = target_weights_bytes + draft_weights_bytes
    target_block_bytes = bytes_per_kv_block(config.hf_config, config.kvcache_block_size, dtype)
    draft_block_bytes = bytes_per_kv_block(config.draft_hf_config, config.kvcache_block_size, dtype) if config.speculative else 0
    combined_block_bytes = target_block_bytes + draft_block_bytes
    if combined_block_bytes <= 0:
        raise RuntimeError("invalid combined KV bytes per logical block")

    target_temp, _ = _profile_one_model_peak(target_model, config.hf_config, config, device, dtype, weights_bytes)
    draft_temp, _ = _profile_one_model_peak(draft_model, config.draft_hf_config, config, device, dtype, weights_bytes) if config.speculative else (0, 0)

    temporary_peak_bytes = max(target_temp, draft_temp) # temporary peak为draft和target model的峰值最大值, 而非二者之和, 因为二者的activations不会同时驻留在显存 
    kv_budget_bytes = budget_bytes - weights_bytes - temporary_peak_bytes
    num_blocks = kv_budget_bytes // combined_block_bytes
    if num_blocks <= 0:
        raise RuntimeError("Fatal: No memory left for paired target/draft KV blocks.")
    
    return KVMemoryProfile(
        total_bytes=total_bytes,
        budget_bytes=budget_bytes,
        reserved_bytes=reserved_bytes,
        weights_bytes=int(weights_bytes),
        temporary_peak_bytes=int(temporary_peak_bytes),
        block_bytes=int(combined_block_bytes),
        num_blocks=int(num_blocks),
        target_block_bytes=int(target_block_bytes),
        draft_block_bytes=int(draft_block_bytes),
        target_weights_bytes=int(target_weights_bytes),
        draft_weights_bytes=int(draft_weights_bytes),
    )



# Real model runner
# 当前版本SD模式仅考虑全局(model-level)共享同一个draft token length(= max_draft_tokens), 不考虑request-level/user-level的自适应draft token length调整
class ModelRunner:
    def __init__(self, config, device:str="cuda", dtype:torch.dtype=torch.bfloat16): # 把model从初始化列表中移除, 因为model固定为Qwen3ForCausalLM
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.sampler = Sampler()
        self.draft_model = None
        self.kv_cache = None
        self.draft_kv_cache = None
        # self._max_draft_tokens = int(config.max_draft_tokens)   # max draft token length, model level
        # self._adaptive_draft = bool(config.adaptive_draft)      # 是否允许自适应调整draft token length # TODO
        # self._effective_draft_tokens = min(2, self._max_draft_tokens) if self._adaptive_draft else self._max_draft_tokens
        # self._acceptance_ema: float | None = None               # 近期接受率的指数平滑结果
        self._drafted_tokens = 0                                # 累计 proposal token 数
        self._accepted_tokens = 0                               # 累计 accepted proposal token 数
        self._verification_rounds = 0                           # 累计有效 SD decode request-round 数

        # 加载target model的metadata
        if self.config.hf_config is None:
            self.config.load_hf_config()
        hf_config = self.config.hf_config
        hf_config.kvcache_block_size = self.config.kvcache_block_size

        # 如果是SD模式则继续加载draft model的metadata
        if self.config.speculative:
            if self.config.draft_hf_config is None:
                self.config.load_hf_config()
            draft_hf_config = self.config.draft_hf_config
            draft_hf_config.kvcache_block_size = self.config.kvcache_block_size
            if int(hf_config.vocab_size) != int(draft_hf_config.vocab_size):
                raise ValueError("Fatal: Target and draft vocab_size differ.")
            if getattr(hf_config, "model_type", None) != getattr(draft_hf_config, "model_type", None):
                raise ValueError("Fatal: Target and draft model_type mismatch in this Qwen3 runtime.")
        

        # step1: 先不绑定kv cache, 单纯构造模型(target和draft)
        self.model = Qwen3ForCausalLM(hf_config, kv_cache = None, backend = self.config.attention_backend).to(device = self.device, dtype = self.dtype)
        self.draft_model = Qwen3ForCausalLM(draft_hf_config, kv_cache=None, backend = self.config.attention_backend).to(device=self.device, dtype=self.dtype) if self.config.speculative else None

        # step2: 加载真实的Qwen3 checkpoint
        self.load_report = load_weights(self.model, self.config.model, strict = True)
        self.model.eval()
        if self.config.speculative:
            self.draft_load_report = load_weights(self.draft_model, self.config.draft_model, strict=True)
            self.draft_model.eval()

        # step3: 进行memory profile, 模型参数驻留后且扣除动态temporary peak之后再估算KV容量
        # self.resource_profile = build_kv_memory_profile(self.config, self.device, self.dtype, self.model)
        self.resource_profile = build_kv_memory_profile(self.config, self.device, self.dtype, self.model, self.draft_model)
        num_blocks = self.resource_profile.num_blocks
        self.config.num_kvcache_blocks = num_blocks
        
        # step4: 创建共享的PagedKVCache并绑定到所有decoder layers
        head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads))
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.config.kvcache_block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.model.set_kv_cache(self.kv_cache)

        if self.config.speculative:
            draft_hf_config = self.config.draft_hf_config
            draft_head_dim = int(getattr(draft_hf_config, "head_dim", draft_hf_config.hidden_size // draft_hf_config.num_attention_heads))
            self.draft_kv_cache = PagedKVCache(
                num_layers=draft_hf_config.num_hidden_layers,
                num_blocks=num_blocks,
                block_size=self.config.kvcache_block_size,
                num_kv_heads=draft_hf_config.num_key_value_heads,
                head_dim=draft_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.draft_model.set_kv_cache(self.draft_kv_cache)

        


    @property
    def speculative_enabled(self) -> bool:
        return bool(self.config.speculative)


    # emit SD相关性能参数, 用于 UI 更新
    def speculative_snapshot(self) -> dict:
        proposed = self._drafted_tokens
        accepted = self._accepted_tokens
        rounds = self._verification_rounds
        return {
            "enabled": self.speculative_enabled,
            "drafted_tokens": proposed,  # 保留原始累计量便于调试
            "accepted_tokens": accepted,
            "verification_rounds": rounds,
            "acceptance_rate": accepted / proposed if proposed else None,
            "avg_acceptance_per_round": accepted / rounds if rounds else None,
        }

    def reset_speculative_statistics(self) -> None:
        self._drafted_tokens = 0
        self._accepted_tokens = 0
        self._verification_rounds = 0


    # 对多个seq准备prefill workload; deprecated
    # 假设一个batch包含3个prefill请求, seq0 = [A, B, C, D, | E, F], seq1 = [a, b, | c, d], seq2 = [|1, 2, 3]; (|右侧代表尚未进入prefix cache的部分)
    # 则执行完prepare_prefill之后:
    #  input_ids = [E, F, c, d, 1, 2, 3]
    #  positions = [4, 5, 2, 3, 0, 1, 2]
    #  slot_mapping = [81, 82, 23, 24, 56, 57, 58]
    #  q_lens = [2, 2, 3]; k_lens = [6, 4, 3]
    #  cu_q = [0, 2, 4, 7]; cu_k = [0, 6, 10, 13]
    def prepare_prefill(self, seqs): 
        input_ids = [] # 真正送给模型计算的 token, 而非完整prompt; 多个seq请求直接拼接而成
        positions = [] # 这些 token 在原始 sequence 中的位置; 多个seq请求直接拼接而成
        slot_mapping = [] # 注意这里仅代表本批prefill最终写入KV cache的slot, 不包含读取的prefix cache对应的slot
        q_lens = []    # 每个seq本次prefill需要真正计算的长度
        k_lens = []    # 每个seq本次attention看到的历史token长度

        for seq in seqs:
            start = seq.num_prefix_cached_tokens # 已经位于prefix cache中的token数目
            end = seq.num_tokens
            tokens = seq.token_ids[start:end]
            pos = list(range(start, end)) # 真正需要prefill的token下标范围

            input_ids.extend(tokens)
            positions.extend(pos)
            slot_mapping.extend(token_to_slot(seq, p) for p in pos)

            q_lens.append(len(tokens))
            k_lens.append(end)

        def prefix_sum(lengths):
            out = [0]
            for x in lengths:
                out.append(out[-1] + x)
            return out

        cu_q = prefix_sum(q_lens) # 导出flashAttention所需要的cumsum形式
        cu_k = prefix_sum(k_lens)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, device=self.device)
        cu_k = torch.tensor(cu_k, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens, default=0),
            max_seqlen_k=max(k_lens, default=0),
            slot_mapping=slot_mapping,
        )
        return input_ids, positions

    # 对多个seq准备decode workload; deprecated
    # 假设一个batch同样包含3个decode请求, seq0 = [A, B, C], seq1 = [a, b, c, d], seq2 = [1, 2];
    # 则执行完prepare_docode之后(假设block_size = 2):
    #  input_ids = [C, d, 2]
    #  positions = [2, 3, 1]
    #  slot_mapping = [60, 71, 12]
    #  context_lens = [3, 4, 2]
    #  block_tables = [[11, 12], [34, 35], [59, -1]], 所有历史token对应的KV Cache block编号
    def prepare_decode(self, seqs):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        block_tables = []

        max_blocks = max(len(seq.block_table) for seq in seqs)

        for seq in seqs:
            pos = seq.num_tokens - 1
            input_ids.append(seq.last_token)
            positions.append(pos)
            slot_mapping.append(token_to_slot(seq, pos))
            context_lens.append(seq.num_tokens)

            table = list(seq.block_table)
            table.extend([-1] * (max_blocks - len(table))) # 末尾补-1, 代表这个seq比较短、后续block用不到
            block_tables.append(table)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=self.device)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions



    # 真正的统一调度接口, 实现了mixed batching; 唯一目的就是利用schedule挑选出来的seq去设置context
    def prepare_batch(self, output: SchedulerOutput) -> tuple[torch.Tensor, torch.Tensor]:
        if not output.items:
            raise ValueError("cannot prepare an empty SchedulerOutput")
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        q_lens: list[int] = []
        context_lens: list[int] = []
        is_prefill: list[bool] = []
        is_sample: list[bool] = []
        block_tables: list[list[int]] = []
        max_blocks = max(len(item.seq.block_table) for item in output.items) # 注意这里的block已经被预分配

        for item in output.items: # type: ScheduledItems
            seq = item.seq
            start = item.start_pos
            end = start + item.num_tokens

            tokens = seq.token_ids[start:end] # 需要送入模型的tokens
            pos = list(range(start, end))

            if len(tokens) != item.num_tokens: # 逻辑上冗余, 但python的切片在越界时不会报错, 因此需要保留这个检测
                raise RuntimeError("scheduler/model-runner token range mismatch")
            
            input_ids.extend(tokens) # 各个seq请求的输入tokens直接拼接
            positions.extend(pos)    # 各个seq请求的tokens绝对位置直接拼接
            slot_mapping.extend(token_to_slot(seq, p) for p in pos) # 各个seq请求的tokens对应的physical slot直接拼接

            q_lens.append(item.num_tokens)      # 每个seq请求本轮输入tokens的数目
            context_lens.append(end)            # 每个seq请求本轮前向传播完毕后,总上下文长度
            is_prefill.append(item.is_prefill)  # 每个seq请求本轮是否处于prefill阶段
            is_sample.append(item.sample_after) # 每个seq请求本轮是否需要采样下一个token

            table = list(seq.block_table)       # 每个seq请求本轮涉及到的block_table, 且长度向max_len对齐, 末尾补-1
            table.extend([-1] * (max_blocks - len(table)))
            block_tables.append(table)

        cu_q = [0]
        for q_len in q_lens:
            cu_q.append(cu_q[-1] + q_len)       # 额外对q_lens求cumsum数组

        input_ids_ret = torch.tensor(input_ids, dtype=torch.long, device=self.device)   # [num_flat_tokens]
        positions_ret = torch.tensor(positions, dtype=torch.long, device=self.device)   # [num_flat_tokens]

        set_context(
            q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor( # [num_seq]
                context_lens, dtype=torch.int32, device=self.device
            ),
            cu_seqlens_q=torch.tensor( # [num_seq + 1]
                cu_q, dtype=torch.int32, device=self.device
            ),
            max_seqlen_q=max(q_lens, default=0),
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.long, device=self.device
            ),
            block_tables=torch.tensor(
                block_tables, dtype=torch.int32, device=self.device
            ),
            is_prefill=torch.tensor( # [num_seq]
                is_prefill, dtype=torch.bool, device=self.device
            ),
            is_sample=torch.tensor( # [num_seq]
                is_sample, dtype=torch.bool, device=self.device
            ),
        )

        return input_ids_ret, positions_ret



    # run in target-only mode, i.e., without speculative decoding
    # 一个 mixed batch 只执行一次 Target forward
    # partial prefill 不产生 token; final prefill / decode 各产生一个 token
    # 返回：{ seq_id: [token_id]}
    @torch.no_grad() # 这个函数里的计算不需要构建 autograd 计算图，也不需要反向传播; 适合纯推理任务
    def run_without_sd(self, output: SchedulerOutput) -> dict[int, list[int]]:
        if not output.items:
            return {}
        try:
            input_ids, positions = self.prepare_batch(output) # shape = [num_flat_tokens]
            # 一个batch全部送入模型进行前向传播 
            hidden_states, _ = self.model(input_ids, positions, output_hidden_states = False) # shape = [num_flat_tokens, hidden_dim]
            
            # 本轮所有 Target 输入都已经真正写入 Target KV
            for item in output.items:
                end = item.start_pos + item.num_tokens
                item.seq.target_kv_computed_tokens = end

            # flat hidden_states 中只抽取需要 sample 的每条 sequence 最后一个 query
            sample_flat_indices: list[int] = []
            sample_items = []
            cursor = 0

            for item in output.items:  # 找出真正需要采样的 sequence
                cursor += item.num_tokens
                if item.sample_after:
                    sample_flat_indices.append(cursor - 1) # hidden_states中需要被采样的token向量的位置; 注意不论是last prefill还是decode, 默认一个seq请求仅有一个token需要采样
                    sample_items.append(item)
            if not sample_items: # 没有采样需求, 说明当前batch全都是partial prefill, 无final prefill或者decode的seq请求
                return {}

            idx = torch.tensor(sample_flat_indices, dtype=torch.long, device=hidden_states.device) # [sampled_seq], 即这一批中is_sample=True的seq请求总数
            sample_hidden = hidden_states.index_select(0, idx) # [sampled_seq, hidden_dim]
            logits = self.model.compute_logits(sample_hidden)  # [sampled_seq, vocab_size]


            if DEBUG:
                for i, item in enumerate(sample_items):
                    if not item.seq.sampling_params.greedy:
                        continue

                values, ids = torch.topk(logits[i].float(), k=2)
                argmax_token = int(logits[i].argmax().item())

                input_pos = item.start_pos + item.num_tokens - 1
                pred_pos = input_pos + 1

                print(
                    f"[BASE] "
                    f"pos={pred_pos} "
                    f"input={item.seq.token_ids[input_pos]} "
                    f"output={argmax_token} "
                    f"argmax={argmax_token} "
                    f"top1_id={ids[0].item()} "
                    f"top1={values[0].item():.6f} "
                    f"top2_id={ids[1].item()} "
                    f"top2={values[1].item():.6f} "
                    f"margin={(values[0] - values[1]).item():.6f}",
                    flush=True,
                )



            temperatures = torch.tensor( # 为每个需要sample的seq设置temperature, shape = [sampled_seq]
                [item.seq.sampling_params.temperature for item in sample_items],
                dtype=logits.dtype,
                device=logits.device,
            )
            topks = torch.tensor(  # shape = [sampled_seq]
                [item.seq.sampling_params.topk for item in sample_items],
                dtype=torch.long, # 下标通常用long; torch的index/gather相关函数通常需要long类型输入
                device=logits.device,
            )
            greedy_mask = torch.tensor( # shape = [sampled_seq]
                [item.seq.sampling_params.greedy for item in sample_items],
                dtype=torch.bool,
                device=logits.device,
            )

            tokens, _ = self.sampler( # 进行采样
                logits=logits,
                temperatures=temperatures,
                topks=topks,
                greedy_mask=greedy_mask,
                isDistributionOut = False
            )

            return { # 第几个item增加了哪个token; 并非所有item都会增加token
                item.seq.seq_id: [int(token)] for item, token in zip(sample_items, tokens.tolist())
            }
        finally:
            # Context 是一次 forward 的动态状态，绝不能泄漏到下一轮
            reset_context()




    # Speculative Decoding mode
    # Prefill: Target 和 Draft 都对相同 token 做 forward, 同时建立两份 KV; 只有 Target 决定最终输出 token
    # Decode: Draft 自回归产生最多 K 个 proposal; Target 一次 ragged forward 并行验证; 返回 accepted proposals + correction/bonus
    # 返回：{seq_id: [token1, token2, ...]};对 decode 而言长度最多 K + 1，最后一个 token 一定是Target 给出的 correction 或 bonus
    @torch.no_grad()
    def run_with_sd(self, output: SchedulerOutput, K: int) -> dict[int, list[int]]:
        """
        假设当前 SchedulerOutput 中包含：
        x 个 prefill item
        y 个 decode item

        整体分为三个阶段：
        Step 1.1: 所有 prefill item 一次性送入 Draft Model, 建立对应的 Draft KV Cache
        Step 1.2: 所有 decode item 以 batch 形式进行 K 轮 Draft decode, 得到每个请求的 d1...dK 及对应 q(x); 最后额外把 dK 再 forward 一次(写入kv cache)
        Step 2: 将 prefill chunk + decode verification segment [pending, d1, ..., dK] 重新组成一个 mixed batch, 一次性送入 Target Model;
                对 decode verification segment 的每一个位置都保留 logits, 从而得到验证 proposal 所需的 p(x) 和最后的 bonus distribution
        Step 3: 对每个 decode item 执行 accept / reject; 返回 accepted draft tokens + correction/bonus
        Return: dict[int, list[int]]

        例如：
            {
                seq0: [first_output],                # final prefill
                seq1: [d1, d2, correction],          # SD reject
                seq2: [d1, d2, d3, d4, bonus],       # all accepted
            }
        """

        if not output.items:
            return {}
        if self.draft_model is None or self.draft_kv_cache is None:
            raise RuntimeError("Fatal: Speculative decoding requires draft model and draft KV cache.")
        if K < 0:  # 防御性检查
            raise ValueError("Fatal: K < 0.")

        results: dict[int, list[int]] = {}
        prefill_items = [item for item in output.items if item.is_prefill]
        decode_items = [item for item in output.items if not item.is_prefill]

        # 为每个 decode seq 的 draft proposal 和 q(x) 预留空间, 并记录每个 decode 起点
        draft_tokens: dict[int, list[int]] = {item.seq.seq_id: [] for item in decode_items}          # proposal, shape = [K]
        draft_probs: dict[int, list[torch.Tensor]] = {item.seq.seq_id: [] for item in decode_items}  # 即Q矩阵, shape = [K, V]
        #target_probs: dict[int, list[torch.Tensor]] = {item.seq.seq_id: [] for item in decode_items} # 即P矩阵, shape = [K + 1, V]
        target_probs: dict[int, torch.Tensor] = {}
        if DEBUG:
            target_raw_logits: dict[int, torch.Tensor] = {} # for debug only
        decode_start: dict[int, int] = {item.seq.seq_id: item.start_pos for item in decode_items}

        # 为保险额外检查draft/target kv cache reservation情况; 相应逻辑理应由schedule处理, 这里是防御性编程
        # ModelRunner 只负责 forward，不负责 block allocation; Scheduler / BlockManager 必须提前 reserve 到 start + K
        #for item in decode_items:
        #    seq = item.seq
        #    last_write_position = item.start_pos + K
        #    logical_block = (last_write_position // seq.block_size)
        #    if logical_block >= len(seq.block_table):
        #        raise RuntimeError("Fatal: Insufficient KV cache reservation for speculative decoding.")

    
        # Step 1.1: 将所有 prefill item 打成一个 batch 送入 Draft Model 进行一轮forward
        if prefill_items:
            prefill_output = SchedulerOutput(prefill_items)
            try:
                input_ids, positions = self.prepare_batch(prefill_output)
                self.draft_model(input_ids, positions, output_hidden_states=False)

                # Draft KV 已经真实写到各 prefill item 的 end
                for item in prefill_items:
                    item.seq.draft_kv_computed_tokens = item.start_pos + item.num_tokens

            finally:
                reset_context()


        # Step 1.2: 所有 decode item 进行 K + 1 轮decode, 前 K 轮采样 + 记录logits向量q(x); 最后1轮不采样
        if decode_items:
            max_blocks = max(len(item.seq.block_table) for item in decode_items)

            # 一共执行 K + 1 次 Draft forwar, 其中前 K 次生成 proposal，最后一次只写KV cache
            for depth in range(K + 1):
                input_token_ids: list[int] = []
                positions: list[int] = []
                slot_mapping: list[int] = []
                context_lens: list[int] = []
                block_tables: list[list[int]] = []

                for item in decode_items: # 打包所有的decode请求
                    seq = item.seq
                    seq_id = seq.seq_id
                    start = decode_start[seq_id]    # 上一轮 pending 的位置
                    position = start + depth        # 本轮KV cache 写入的frontier position
                    token = seq.token_ids[start] if depth == 0 else draft_tokens[seq_id][depth - 1]
                
                    input_token_ids.append(token)   # 每个seq请求本轮需要输入模型的token_id
                    positions.append(position)      # 每个token的绝对位置
                    slot_mapping.append(token_to_slot(seq, position)) # 输入模型的token_id对应的物理kv cache索引, 用于kv cache写入
                    context_lens.append(position + 1) # 每个seq请求已经生成的token长度, 用于attention计算时gather历史kv cache
                    table = list(seq.block_table)    
                    table.extend([-1] * (max_blocks - len(table)))
                    block_tables.append(table)      # 已分配block id列表, -1用于尾部对齐、代表尚未分配的block

                batch_size = len(decode_items)

                try: # 对y个decode请求打包, set_context然后进行forward
                    set_context(
                        q_lens=torch.ones(batch_size, dtype=torch.int32, device=self.device), # y个1
                        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
                        cu_seqlens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=self.device),
                        max_seqlen_q=1,
                        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
                        block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
                        is_prefill=torch.zeros(batch_size, dtype=torch.bool, device=self.device), # 所有seq都不是prefill
                        is_sample=torch.full((batch_size,), depth < K, dtype=torch.bool, device=self.device), # 仅前 K 轮需要采样
                    )

                    # 进行forward
                    input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
                    positions_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
                    hidden_states, _ = self.draft_model(input_ids, positions_tensor, output_hidden_states=False)

                    for item in decode_items: # 当前输入 token 已经真正写入 Draft KV; 这里遵循右开区间设定, 位置指向下一个未被写入的pos
                        item.seq.draft_kv_computed_tokens = decode_start[item.seq.seq_id] + depth + 1
                    
                    # 只有前 K 次需要采样
                    if depth == K:
                        continue
                    
                    # 准备采样参数
                    logits = self.draft_model.compute_logits(hidden_states) # [y, vocab_size]
                    temperatures = torch.tensor([item.seq.sampling_params.temperature for item in decode_items], dtype=logits.dtype, device=logits.device)
                    topks = torch.tensor([item.seq.sampling_params.topk for item in decode_items], dtype=torch.long, device=logits.device)
                    greedy_mask = torch.tensor([item.seq.sampling_params.greedy for item in decode_items], dtype=torch.bool, device=logits.device)

                    # 正式采样并返回q(x)
                    tokens, draft_distributions = self.sampler(logits=logits, temperatures=temperatures, topks=topks, greedy_mask=greedy_mask, isDistributionOut=True)

                    for i in range(len(decode_items)): # 获取decode batch中每个seq本轮生成的token
                        item = decode_items[i]
                        seq = item.seq
                        seq_id = seq.seq_id
                        draft_tokens[seq_id].append(tokens[i].item())
                        draft_probs[seq_id].append(draft_distributions[i])

                finally: # 每轮batched decoding结束都要复位context
                    reset_context()


        # Step 2: 将step1输出的decode序列和原始的prefill item再拼成一个mixed batch, 送入target model进行一次forward
        input_ids_flat: list[int] = []
        positions_flat: list[int] = []
        slot_mapping_flat: list[int] = []
        q_lens: list[int] = []
        context_lens: list[int] = []
        is_prefill: list[bool] = []
        is_sample: list[bool] = []
        block_tables: list[list[int]] = []      # [num_seq]
        flat_sample_mask: list[bool] = []       # [num_flat_tokens]
        flat_sample_mask_cumsum: list[int] = [0] # 每个item对应的sample logits行前缀和
        max_blocks = max(len(item.seq.block_table) for item in output.items)

        # 按 SchedulerOutput 原始 item 顺序重新构造 mixed batch
        for item in output.items:
            seq = item.seq
            seq_id = seq.seq_id
            
            if item.is_prefill: # 处理原始 prefill item
                start = item.start_pos
                end = start + item.num_tokens
                tokens = seq.token_ids[start:end]
                positions = list(range(start, end))
                input_ids_flat.extend(tokens)       # 注意这里是直接flatten之后跨seq请求拼接
                positions_flat.extend(positions)
                slot_mapping_flat.extend(token_to_slot(seq, position) for position in positions)

                q_lens.append(item.num_tokens)      # 这里是每个seq请求对应一个参数
                context_lens.append(end)
                is_prefill.append(True)
                is_sample.append(item.sample_after)
                table = list(seq.block_table)
                table.extend([-1] * (max_blocks - len(table)))
                block_tables.append(table)
                
                flat_sample_mask.extend([False] * item.num_tokens)
                sample_count = 1 if item.sample_after else 0
                if item.sample_after:
                    flat_sample_mask[-1] = True
                flat_sample_mask_cumsum.append(flat_sample_mask_cumsum[-1] + sample_count)

            else: # 处理 draft decode item
                start = decode_start[seq_id]
                verify_tokens = [seq.token_ids[start], *draft_tokens[seq_id]] # [pending_token, draft decode token]
                verify_len = len(verify_tokens)

                positions = list(range(start, start + verify_len))
                input_ids_flat.extend(verify_tokens)    # input_id、position、slot_mapping均为展平后的向量(token-level metadata), 长度等于num_total_flat_tokens
                positions_flat.extend(positions)
                slot_mapping_flat.extend(token_to_slot(seq, position) for position in positions)

                q_lens.append(verify_len)               # 这几个为seq-level metadata, 长度为y
                context_lens.append(start + verify_len)
                is_prefill.append(False)
                is_sample.append(True)
                table = list(seq.block_table)
                table.extend([-1] * (max_blocks - len(table)))
                block_tables.append(table)
                
                flat_sample_mask.extend([True] * verify_len)
                flat_sample_mask_cumsum.append(flat_sample_mask_cumsum[-1] + verify_len)
                
        # q_lens -> cu_seqlens_q
        cu_q = [0]
        for q_len in q_lens:
            cu_q.append(cu_q[-1] + q_len)

        try:
            set_context( # 准备context(metadata)
                q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),
                context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
                cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=self.device),
                max_seqlen_q=max(q_lens),
                slot_mapping=torch.tensor(slot_mapping_flat, dtype=torch.long, device=self.device),
                block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
                is_prefill=torch.tensor(is_prefill, dtype=torch.bool, device=self.device),
                is_sample=torch.tensor(is_sample, dtype=torch.bool, device=self.device),
            )

            # 准备input数据
            input_ids = torch.tensor(input_ids_flat, dtype=torch.long, device=self.device)
            positions = torch.tensor(positions_flat, dtype=torch.long, device=self.device)

            # 核心: 所有 prefill + 所有 decode verify 只执行一次 Target forward
            target_hidden, _ = self.model(input_ids, positions, output_hidden_states=False)

            # Target KV 已经按照各自 query segment 正常写入。
            for item, end in zip(output.items, context_lens):
                item.seq.target_kv_computed_tokens = end

            sample_indices = [idx for idx, flag in enumerate(flat_sample_mask) if flag]
            sample_hidden = target_hidden.index_select(0, torch.tensor(sample_indices, dtype=torch.long, device=self.device))
            target_logits = self.model.compute_logits(sample_hidden) # [num_sample_tokens, vocab_size]
            
            # 准备采样参数
            sample_temperatures: list[float] = []
            sample_topks: list[int] = []
            sample_greedy_mask: list[bool] = []
            for item_idx in range(len(output.items)):
                item = output.items[item_idx]
                sample_count = flat_sample_mask_cumsum[item_idx + 1] - flat_sample_mask_cumsum[item_idx]
                sample_temperatures.extend([item.seq.sampling_params.temperature] * sample_count)
                sample_topks.extend([item.seq.sampling_params.topk] * sample_count)
                sample_greedy_mask.extend([item.seq.sampling_params.greedy] * sample_count)
            temperatures = torch.tensor(sample_temperatures, dtype=target_logits.dtype, device=target_logits.device)
            topks = torch.tensor(sample_topks, dtype=torch.long, device=target_logits.device)
            greedy_mask = torch.tensor(sample_greedy_mask, dtype=torch.bool, device=target_logits.device)

            # 正式采样并返回p(x)
            target_tokens, target_distributions = self.sampler(logits=target_logits, temperatures=temperatures, topks=topks, greedy_mask=greedy_mask, isDistributionOut=True)

            # 根据flat_sample_mask_cumsum将全局P矩阵切回每个decode seq
            for item_idx in range(len(output.items)):
                item = output.items[item_idx]
                left, right = flat_sample_mask_cumsum[item_idx], flat_sample_mask_cumsum[item_idx + 1]
                if item.is_prefill: # 如果是final prefill, 直接记录结果
                    if item.sample_after:
                        results[item.seq.seq_id] = [int(target_tokens[left].item())] 
                    continue
                target_probs[item.seq.seq_id] = target_distributions[left:right] # 如果是decode, 则记录P矩阵
                if DEBUG:
                    target_raw_logits[item.seq.seq_id] = target_logits[left:right] # for debug only

        finally:
            reset_context()


        # Step 3: 直接按 flat_sample_mask 顺序消费 target_logits
        batch_proposed = 0
        batch_accepted = 0
        batch_rounds = 0

        for item in decode_items:
            seq = item.seq
            seq_id = seq.seq_id

            proposals = draft_tokens[seq_id]    # [K]
            P = target_probs[seq_id]            # [K + 1, V]
            Q = (torch.stack(draft_probs[seq_id], dim=0) if draft_probs[seq_id] else P.new_empty((0, P.shape[1])))  # [K, V], K >= 0

            eos_token_id = (-1 if seq.sampling_params.ignore_eos else self.config.eos_token_id)
            final_tokens, accepted_count = speculative_verify(proposals, Q, P, eos_token_id)
            results[seq_id] = final_tokens

            batch_proposed += len(proposals)
            batch_accepted += accepted_count
            batch_rounds += 1

            if DEBUG:
                raw_logits = target_raw_logits[seq_id]
                for i, token in enumerate(final_tokens):
                    logits_i = raw_logits[i].float()
                    values, ids = torch.topk(logits_i, k=2)
                    argmax_token = int(raw_logits[i].argmax().item())
                    if seq.sampling_params.greedy:
                        assert token == argmax_token, (
                            f"Greedy SD mismatch: pos={item.start_pos + i + 1}, "
                            f"output={token}, target_argmax={argmax_token}"
                        )
                    proposal = proposals[i] if i < len(proposals) else None
                    accepted = i < accepted_count
                    input_token = (seq.token_ids[item.start_pos] if i == 0 else proposals[i - 1])

                    print(
                        f"[SD] "
                        f"K={K} "
                        f"pos={item.start_pos + i + 1} "
                        f"input={input_token} "
                        f"proposal={proposal} "
                        f"accepted={accepted} "
                        f"output={token} "
                        f"argmax={argmax_token} "
                        f"top1_id={ids[0].item()} "
                        f"top1={values[0].item():.6f} "
                        f"top2_id={ids[1].item()} "
                        f"top2={values[1].item():.6f} "
                        f"margin={(values[0] - values[1]).item():.6f}",
                        flush=True,
                    )

        # 更新全局SD statistics
        self._drafted_tokens += batch_proposed
        self._accepted_tokens += batch_accepted
        self._verification_rounds += batch_rounds

        return results




    # ModelRunner 唯一对外 forward 接口, 根据是否为SD模式分别调用run_w/wo_sd
    # K代表SD模式下draft decode最多允许提出的proposal token数目
    def run(self, output: SchedulerOutput, K: int) -> dict[int, list[int]]:
        if self.config.speculative:
            return self.run_with_sd(output, K)
        return self.run_without_sd(output)
