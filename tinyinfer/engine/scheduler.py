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

    def describe_output(output): # for debug only
        return [
            {
                "seq_id": x.seq.seq_id,
                "phase": "prefill" if x.is_prefill else "decode",
                "start": x.start_pos,
                "n": x.num_tokens,
                "sample": x.sample_after,
            }
            for x in output.items
        ]

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



    # 优先prefill, 其次decode, 且一个batch里面仅有prefill或者decode
    def schedule_separated(self) -> tuple[list[Sequence], bool]:
        if trace_enabled("TINYINFER_TRACE_SCHED"):
            print(
                "[sched]",
                "waiting=", len(self.waiting),
                "waiting_ids=", [s.seq_id for s in self.waiting],
                "running=", len(self.running),
                "running_ids=", [s.seq_id for s in self.running],
            )
        if trace_enabled("TINYINFER_CHECK_KV"):
            self.block_manager.check_consistency()

        scheduled: list[Sequence] = []
        token_budget = self.max_num_batched_tokens # 一轮调度中最多支持多少tokens
        # ------------------------------------------------------------
        # Phase A: admit waiting requests for prefill
        # ------------------------------------------------------------
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0] # 所有需要prefill的seq必然位于waiting队列中, 这里采用的是FIFO队列, 建模比较简单, 后续要改调度方案
            prefill_tokens = seq.num_tokens - seq.num_prefix_cached_tokens # 真正需要prefill的token数, 需要减去已经位于prefix cache中的token数目
            if prefill_tokens > token_budget:           # 确保本轮推理预算足够
                break
            if not self.block_manager.try_allocate_with_prefix_cache(seq): # 为本次请求分配prefix cache, 更新seq.block_table和blockManager中的block池
                break
            self.waiting.popleft() # 通过检查, 放入running列表
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = True
            seq.mark_scheduled(prefill_tokens)
            self.running.append(seq)
            scheduled.append(seq)
            token_budget -= prefill_tokens # 从总budget中扣除已经用掉的部分
        if scheduled:
            return scheduled, True
        # ------------------------------------------------------------
        # Phase B: one-token decode for running requests
        # ------------------------------------------------------------
        for seq in list(self.running): # 只有当本轮没有成功调度任何 prefill 时，才会走到 decode
            if token_budget <= 0:
                break
            # 已达到最大模型长度，不应该再送进 ModelRunner
            if seq.num_tokens >= self.max_model_len:
                continue
            seq.is_prefill = False
            seq.mark_scheduled(1)
            scheduled.append(seq)
            token_budget -= 1
        return scheduled, False


    # 当前的schedule会额外负责每个seq请求后续block的分配: 
    #  对于prefill会预分配所有prompt token对应的block(不论admit时是否已经在prefix cache中)
    #  对于decode, 如果有需求, 会预分配下一个block
    def schedule(self) -> SchedulerOutput:
        items: list[ScheduledItem] = []
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs

        # Phase A: decode first; 注意当前通常num_computed_token比num_tokens少1, 因为有1个token是上一轮decode得到的、尚未算出KV cache
        for seq in list(self.running):
            if token_budget <= 0 or seq_budget <= 0: # token总数/seq总数达到一批推理的容量上限
                break
            if seq.num_tokens >= self.max_model_len: # 本条seq的token总数已经达到上限
                continue
            if not seq.needs_decode: # 确认当前需要继续decode
                continue

            start = seq.num_computed_tokens
            if not self.block_manager.ensure_capacity_for_token_position(seq, start): # 预分配block
                continue

            seq.mark_scheduled(1)
            items.append(
                ScheduledItem(
                    seq=seq,
                    start_pos=start,
                    num_tokens=1,
                    is_prefill=False,
                    sample_after=True,
                )
            )
            token_budget -= 1
            seq_budget -= 1

        # Phase B: then chunked prefill with remaining budget; 注意只有一个seq的chunked prefill彻底完成, 其才会离开waiting list
        waiting_count = len(self.waiting)

        for _ in range(waiting_count):
            if token_budget <= 0 or seq_budget <= 0:
                break
            seq = self.waiting[0] # 从头部获取第一个waiting的seq

            # First admission: prefix lookup + block allocation.
            if not seq.block_table: # block_table为空说明必然为首次进行prefill, 因此需要admit
                if not self.block_manager.try_admit_with_prefix_cache(seq):
                    break

            remaining = seq.num_prompt_tokens - seq.num_computed_tokens
            if remaining <= 0:
                raise RuntimeError("waiting prefill has no remaining prompt tokens")

            chunk = min(remaining, token_budget) # 这次chunk的大小
            start = seq.num_computed_tokens
            end = start + chunk

            # try_admit_with_prefix_cache() 当前已为完整 prompt 建好 block table, 因此这里只做防御检查。
            #if not self.block_manager.ensure_capacity_for_token_position(seq, end - 1):
            #    break

            sample_after = end == seq.num_prompt_tokens
            seq.mark_scheduled(chunk)

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

            # partial prefill 做 round-robin优先级调度, 最近被处理过的seq被放到 waiting 尾部。
            self.waiting.rotate(-1)

        return SchedulerOutput(items)


    def postprocess(self, output: SchedulerOutput, sampled_tokens: dict[int, int]):
        finished = []
        prefill_completed = []

        for item in output.items:
            seq = item.seq
        
            # 1. 本轮 scheduled token 的 KV 已经完成。
            seq.num_computed_tokens += item.num_tokens
            seq.num_scheduled_tokens = 0

            # 2. 到新的 computed frontier 为止，注册 newly-full blocks
            self.block_manager.cache_computed_full_blocks(seq, upto_token=seq.num_computed_tokens)

            # 3. partial prefill 不采样，直接结束本 item
            if not item.sample_after: # 此时sample_tokens实际为None
                continue
            else:
                token_id = sampled_tokens[seq.seq_id]
                seq.append_token(token_id) # 这里让num_tokens + 1, 但num_computed_tokens没变

            # 4. 采样后检查停止条件。
            if seq.should_stop(self.eos_token_id, self.max_model_len):
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)
                continue

            # 5. 如果刚完成 prompt，则进入 running decode 集合
            if item.is_prefill: # 走到这里的prefill必然是final prefill
                prefill_completed.append(seq)
                seq.status = SequenceStatus.RUNNING

        # waiting 中删除完成 prefill 或 finished 的请求。
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

        # 最后释放 request ownership；persistent cached block 不会被 reset。
        for seq in finished:
            self.block_manager.free(seq)







