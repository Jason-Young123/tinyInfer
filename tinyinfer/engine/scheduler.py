from collections import deque
from dataclasses import dataclass

from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence, SequenceStatus, ScheduledItem, SchedulerOutput
from tinyinfer.utils.debug import trace_enabled
from tinyinfer.engine.block_manager import BlockManager


# 调度器, 本身不负责运行前向传播, 而是选择合适的sequence给ModelRunner进行前向传播
# 待改进: 加入真正的Mixed Batching; 改进调度的公平性


class Scheduler:
    def __init__(self, config: Config):
        # 一轮最多允许多少条 Sequence 同时处于 running。
        # Day01 暂时只真正使用 max_num_seqs；
        # max_num_batched_tokens 会在后续 token-budget / chunked prefill 中使用。
        self.max_num_seqs = config.max_num_seqs                     # 同一批次最多同步推进多少条请求
        self.max_num_batched_tokens = config.max_num_batched_tokens # 同一批次所有请求合起来最多不能超过多少tokens
        self.max_model_len = config.max_model_len                   # 单条 Sequence 最多允许多长的总上下文
        self.max_parallel_verify_tokens = config.max_draft_tokens + 1 if config.speculative else 1 # 一轮最多允许并行验证多少个token( = K + 1)
        self.eos_token_id = config.eos_token_id

        # waiting：已经进入推理引擎，但还没有获得运行资格的请求。
        # running：已经被 Scheduler 选中，正在参与 prefill / decode 的请求。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
        )

    def add(self, seq: Sequence): # 新请求首先进入 waiting queue
        self.waiting.append(seq)

    def is_finished(self) -> bool: # waiting 和 running 都为空，说明整个引擎没有未完成请求
        return not self.waiting and not self.running


    def check_consistency(self): # for debug only
        waiting_ids = {s.seq_id for s in self.waiting}
        running_ids = {s.seq_id for s in self.running}
        assert waiting_ids.isdisjoint(running_ids)
        for seq in self.waiting:
            assert seq.status is SequenceStatus.WAITING
            assert seq.num_computed_tokens <= seq.num_prompt_tokens
        for seq in self.running:
            assert seq.status is SequenceStatus.RUNNING
            assert seq.prompt_computed


    def set_max_draft_tokens(self, value: int): # 有UI动态修改max_draft_token_length
        if not 1 <= value <= 32:
            raise ValueError("Fatal: max_draft_tokens out of range [1, 32]")
        self.max_parallel_verify_tokens = value + 1


    # 当前的schedule会额外负责每个seq请求后续block的分配: 
    #  对于prefill会预分配所有prompt token对应的block(不论admit时是否已经在prefix cache中)
    #  对于decode, 如果有需求, 会预分配所需block(可能是多个, 比如在SD模式)
    def schedule(self) -> tuple[SchedulerOutput, int]:
        items: list[ScheduledItem] = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # waiting非空时，Phase A至少给prefill预留1个seq预算和相应的token预算
        decode_seq_budget = seq_budget if not self.waiting else max(0, seq_budget - 1)
        decode_token_budget = token_budget if not self.waiting else max(0, int(0.8 * token_budget))

        # Phase A: decode first with round-robin; 注意当前通常num_computed_token比num_tokens少1, 因为有1个token是上一轮decode得到的、尚未算出KV cache
        running_count = len(self.running)
        # 获取当前批次所有decode请求的公共预算大小
        parallel_verify_tokens = self.max_parallel_verify_tokens # budget constraint1: 在SD约束下的每轮最多可以同步并行验证多少个token
        for seq in self.running: # 三种情况下直接忽略: (1)不需要decode; (2)总context无余量; (3)decode阶段无余量
            if (not seq.needs_decode 
                or seq.num_tokens >= self.max_model_len 
                or seq.num_completion_tokens >= seq.sampling_params.max_tokens
            ):
                continue
            remaining_context = self.max_model_len - seq.num_tokens # budget constraint2: 总seq最多还能生成多少个token
            remaining_output = seq.sampling_params.max_tokens - seq.num_completion_tokens # budget constraint3: decode阶段最多还能生成多少个token
            parallel_verify_tokens_tmp = min(self.max_parallel_verify_tokens, remaining_context, remaining_output) # 至少为1
            parallel_verify_tokens = min(parallel_verify_tokens_tmp, parallel_verify_tokens)
        if decode_token_budget > 0: # 极端情况(K很大的时候)
            parallel_verify_tokens = min(parallel_verify_tokens, decode_token_budget)

        for _ in range(running_count):
            if decode_token_budget < parallel_verify_tokens or decode_seq_budget <= 0:  # 只要waiting-list不为空, 则至少预留一个slot给prefill
                break
            seq = self.running[0]
            
            if (# 三种情况下直接忽略: (1)不需要decode; (2)总context无余量; (3)decode阶段无余量; 防御性编程, 理论上不会发生, 因为相应seq已经在上一轮由Postprocess移除了
                seq.num_tokens >= self.max_model_len
                or seq.num_completion_tokens >= seq.sampling_params.max_tokens
                or not seq.needs_decode
            ):
                self.running.rotate(-1)
                continue

            start = seq.num_computed_tokens
            if not self.block_manager.ensure_capacity_for_token_position(seq, start, parallel_verify_tokens): # 预分配block
                self.running.rotate(-1)
                continue

            seq.mark_scheduled(1)
            items.append(
                ScheduledItem(
                    seq=seq,
                    start_pos=start,
                    num_tokens=1,
                    is_prefill=False,
                    sample_after=True, # 默认decode都是需要sample的, 但SD模式下draft/target对于decode seq每一个位置是否采样还依赖model_runner中的逻辑
                )
            )
            token_budget -= parallel_verify_tokens
            seq_budget -= 1
            decode_token_budget -= parallel_verify_tokens
            decode_seq_budget -= 1

            # 最近被decode调度过的seq放到running尾部
            self.running.rotate(-1)

        # Phase B: then chunked prefill with remaining budget; 注意只有一个seq的chunked prefill彻底完成, 其才会离开waiting list
        waiting_count = len(self.waiting)
        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            seq = self.waiting[0] # 从头部获取第一个waiting的seq

            # First admission: prefix lookup + block allocation.
            # 注意admission会分配后续prefill所需的全部block, 因此之后rest prefill轮次无需再allocate block
            if not seq.block_table: # block_table为空说明必然为首次进行prefill, 因此需要admit
                if not self.block_manager.try_admit_with_prefix_cache(seq):
                    self.waiting.rotate(-1)
                    continue

            remaining = seq.num_prompt_tokens - seq.num_computed_tokens
            if remaining <= 0:
                raise RuntimeError("Fatal: Waiting prefill has no remaining prompt tokens.")

            chunk = min(remaining, token_budget) # 这次chunk的大小 # TODO: 这里应该设置chunk大小上限, 避免一个请求占满预算
            start = seq.num_computed_tokens
            end = start + chunk

            sample_after = end == seq.num_prompt_tokens
            seq.mark_scheduled(chunk) # 开始调度一刻

            items.append(
                ScheduledItem(
                    seq=seq,
                    start_pos=start,
                    num_tokens=chunk,
                    is_prefill=True,
                    sample_after=sample_after,
                )
            )

            token_budget -= chunk
            seq_budget -= 1

            # partial prefill 做 round-robin优先级调度, 最近被处理过的seq被放到 waiting 尾部
            self.waiting.rotate(-1)

        return SchedulerOutput(items), parallel_verify_tokens - 1 # K = parallel_verify_tokens - 1


    def postprocess(self, output: SchedulerOutput, sampled_tokens: dict[int, list[int]]):
        finished = []
        prefill_completed = []

        for item in output.items:
            seq = item.seq
            seq.num_scheduled_tokens = 0

            # 1. Prefill
            if item.is_prefill: # 对于prefill逻辑更加简单, 不涉及kv cache rollback
                seq.num_computed_tokens = item.start_pos + item.num_tokens
                if not item.sample_after: # partial prefill不产生新token
                    self.block_manager.cache_computed_full_blocks(seq, upto_token=seq.num_computed_tokens)
                    continue

                tokens = sampled_tokens[seq.seq_id]# final prefill只返回1个Target采样token 
                if len(tokens) != 1:
                    raise RuntimeError("Fatal: Final prefill must return exactly one token.")
                seq.append_token(tokens[0]) # 这里会让num_tokens + 1

                self.block_manager.cache_computed_full_blocks(seq, upto_token=seq.num_computed_tokens)

            # 2. Decode
            else:
                tokens = sampled_tokens[seq.seq_id]
                if not tokens:
                    raise RuntimeError("Fatal: Decode must return at least one token.")
                
                seq.append_token(tokens) # 在SD场景下一个seq可能一步添加多个token

                # rollback
                committed_frontier = item.start_pos + len(tokens) # 计算真正需要保留的KV cache frontier
                seq.target_kv_computed_tokens = committed_frontier
                seq.draft_kv_computed_tokens = committed_frontier
                seq.num_computed_tokens = committed_frontier

                # 必须rollback之后才能注册full block，避免把speculative尾部误注册成prefix cache
                self.block_manager.cache_computed_full_blocks(seq, upto_token=seq.num_computed_tokens)

            # 3. 检查停止条件
            if seq.should_stop(self.eos_token_id, self.max_model_len):
                seq.mark_finished()
                finished.append(seq)
                continue

            # final prefill完成后进入running decode集合
            if item.is_prefill:
                prefill_completed.append(seq)
                seq.status = SequenceStatus.RUNNING


        # waiting 中删除完成 prefill 或 finished 的请求
        remove_ids = { seq.seq_id for seq in prefill_completed + finished}
        if remove_ids: # waiting list中仅保留原本存在且当前不在remove_ids中的seq
            self.waiting = deque(seq for seq in self.waiting if seq.seq_id not in remove_ids)

        # 新完成 prefill 的 seq 开始进入 running decode
        for seq in prefill_completed:
            self.running.append(seq)

        # running 中删除 finished
        if finished:
            done_ids = {seq.seq_id for seq in finished}
            self.running = deque(seq for seq in self.running if seq.seq_id not in done_ids)

        # 最后释放 request ownership；persistent cached block 不会被 reset
        for seq in finished:
            self.block_manager.free(seq)







