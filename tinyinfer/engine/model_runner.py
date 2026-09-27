import torch
from tinyinfer.utils.context import set_context, reset_context


def token_to_slot(seq, token_position: int) -> int:
    block_size = seq.block_size
    logical_block = token_position // block_size
    offset = token_position % block_size
    physical_block = seq.block_table[logical_block]
    return physical_block * block_size + offset




class ToyModelRunner:
    """Control-plane-only runner used in Day01/02.

    It does not run a neural network. For every sequence it simply emits
    `(last_token + 1) % 1000` so that the engine loop can be tested.
    """

    def run(self, sequences, is_prefill: bool): # 当前阶段不论是不是prefill都默认在末尾添加 id+1
        next_tokens = []
        for seq in sequences:
            next_tokens.append((seq.last_token + 1) % 1000)
        return next_tokens



class ModelRunner:
    """Real model runner"""
    def __init__(self, config, model=None, device="cuda"):
        self.config = config
        self.model = model
        self.device = torch.device(device)

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


    
    def run(self, seqs, is_prefill: bool):
        try:
            if is_prefill:
                input_ids, positions = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)

            logits = self.model.compute_logits(input_ids, positions)
            temperatures = torch.tensor(
                [seq.sampling_params.temperature for seq in seqs],
                dtype=torch.float32,
                device=logits.device,
            )
            return self.model.sample(logits, temperatures).tolist()
        finally:
            reset_context()



