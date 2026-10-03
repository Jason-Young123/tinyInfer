import torch
from tinyinfer.utils.context import set_context, reset_context
from tinyinfer.engine.sequence import SchedulerOutput
from tinyinfer.layers.attention import PagedKVCache
from tinyinfer.layers.sampler import Sampler
from tinyinfer.models.qwen3 import Qwen3ForCausalLM
from tinyinfer.utils.loader import load_weights


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

# 模型已经驻留 GPU 后，再根据当前 reserved memory 估算 KV block 数
def estimate_num_kv_blocks(
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    if device.type != "cuda":
        # CPU 单测不做显存 profile
        raise ValueError("Invalid device: cpu; Required: cuda")
        return 0

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(total * config.gpu_memory_utilization)
    reserved = torch.cuda.memory_reserved(device)
    available_for_kv = max(0, budget - reserved)

    per_block = bytes_per_kv_block(config.hf_config, config.kvcache_block_size, dtype)
    if per_block <= 0:
        raise RuntimeError("invalid KV bytes per block")

    num_blocks = available_for_kv // per_block
    if num_blocks <= 0:
        raise RuntimeError("no memory left for KV cache under gpu_memory_utilization")
    return int(num_blocks)








class ToyModelRunner:
    def run1(self, sequences, is_prefill: bool): # 当前阶段不论是不是prefill都默认在末尾添加 id+1
        next_tokens = []
        for seq in sequences:
            next_tokens.append((seq.last_token + 1) % 1000)
        return next_tokens

    def run(self, output: SchedulerOutput) -> dict[int, int]:
        result = {}
        for item in output.items:
            if item.sample_after:
                result[item.seq.seq_id] = (item.seq.last_token + 1) % 1000
        return result


# Real model runner
class ModelRunner:
    def __init__(self, config, device:str="cuda", dtype:torch.dtype=torch.bfloat16): # 把model从初始化列表中移除, 因为model固定为Qwen3ForCausalLM
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.sampler = Sampler()
        
        if self.config.hf_config is None:
            self.config.load_hf_config()
        hf_config = self.config.hf_config
        hf_config.kvcache_block_size = self.config.kvcache_block_size

        # step1: 先不绑定kv cache, 单纯构造模型
        self.model = Qwen3ForCausalLM(hf_config, kv_cache = None).to(device = self.device, dtype = self.dtype)

        # step2: 加载真实的Qwen3 checkpoint
        report = load_weights(self.model, self.config.model, strict = True)
        self.load_report = report
        self.model.eval()

        # step3: 模型参数驻留后再估算KV容量
        num_blocks = estimate_num_kv_blocks(self.config, self.device, self.dtype)
        self.config.num_kvcache_blocks = num_blocks
        head_dim = int(
            getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
        )

        # step4: 创建唯一共享的PagedKVCache
        self.kv_cache = PagedKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.config.kvcache_block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # step5: 绑定到所有 decoder layers, 对应到所有attn
        self.model.set_kv_cache(self.kv_cache)



    # 对多个seq准备prefill workload
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

    # 对多个seq准备decode workload
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

    # 真正的统一调度接口, 实现了mixed batching
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

        input_ids_ret = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions_ret = torch.tensor(positions, dtype=torch.long, device=self.device)

        set_context(
            q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(
                context_lens, dtype=torch.int32, device=self.device
            ),
            cu_seqlens_q=torch.tensor(
                cu_q, dtype=torch.int32, device=self.device
            ),
            max_seqlen_q=max(q_lens, default=0),
            slot_mapping=torch.tensor(
                slot_mapping, dtype=torch.long, device=self.device
            ),
            block_tables=torch.tensor(
                block_tables, dtype=torch.int32, device=self.device
            ),
            is_prefill=torch.tensor(
                is_prefill, dtype=torch.bool, device=self.device
            ),
            is_sample=torch.tensor(
                is_sample, dtype=torch.bool, device=self.device
            ),
        )

        return input_ids_ret, positions_ret



    @torch.no_grad() # 这个函数里的计算不需要构建 autograd 计算图，也不需要反向传播; 适合纯推理任务
    def run(self, output: SchedulerOutput) -> dict[int, int]:
        try:
            input_ids, positions = self.prepare_batch(output) # shape = [num_flat_tokens]
            # 一个batch全部送入模型进行前向传播 
            hidden_states, _ = self.model(input_ids, positions, output_hidden_states = False) # shape = [num_flat_tokens, hidden_dim]

            # flat hidden_states 中只抽取需要 sample 的每条 sequence 最后一个 query
            sample_flat_indices: list[int] = []
            sample_items = []
            cursor = 0

            for item in output.items:
                cursor += item.num_tokens
                if item.sample_after:
                    sample_flat_indices.append(cursor - 1) # hidden_states中需要被采样的token向量的位置
                    sample_items.append(item)
            if not sample_items: # 没有采样需求, 说明当前batch全都是partial prefill, 无final prefill或者decode的seq请求
                return {}

            idx = torch.tensor(sample_flat_indices, dtype=torch.long, device=hidden_states.device) # [sampled_seq], 即这一批中is_sample=True的seq请求总数
            sample_hidden = hidden_states.index_select(0, idx) # [sampled_seq, hidden_dim]
            logits = self.model.compute_logits(sample_hidden)  # [sampled_seq, vocab_size]

            temperatures = torch.tensor( # 为每个需要sample的seq设置temperature, shape = [sampled_seq]
                [item.seq.sampling_params.temperature for item in sample_items],
                dtype=logits.dtype,
                device=logits.device,
            )
            greedy_mask = torch.tensor( # shape = [sampled_seq]
                [item.seq.sampling_params.greedy for item in sample_items],
                dtype=torch.bool,
                device=logits.device,
            )

            tokens = self.sampler(logits, temperatures, greedy_mask)

            return { # 第几个item增加了哪个token; 并非所有item都会增加token
                item.seq.seq_id: int(token) for item, token in zip(sample_items, tokens.tolist())
            }
        finally:
            # Context 是一次 forward 的动态状态，绝不能泄漏到下一轮
            reset_context()



