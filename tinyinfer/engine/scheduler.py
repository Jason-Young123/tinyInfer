from collections import deque

from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence, SequenceStatus
from tinyinfer.utils import trace_enabled


# 调度器, 本身不负责运行前向传播, 而是选择合适的sequence给ModelRunner进行前向传播

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

    def add(self, seq: Sequence):
        """新请求首先进入 waiting queue。"""
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        """waiting 和 running 都为空，说明整个引擎没有未完成请求。"""
        return not self.waiting and not self.running

    def schedule_naive(self) -> tuple[list[Sequence], bool]: # 核心逻辑: 筛选合适的sequence给ModelRuner进行前向传播
        """
        决定本轮让哪些 Sequence 执行。最朴素版本, 仅考虑max_num_seqs, 忽略max_num_batched_tokens
        Day01 策略非常简单：
        1. 只要 running 还有空位，就按 FIFO 从 waiting 搬进去；
        2. 当前所有 running Sequence 一起交给 ModelRunner；
        3. 返回本轮是否包含 prefill 请求。
        """
        # 把 waiting 中的请求尽可能填入 running。
        # deque.popleft() 表示 FIFO：先来的请求先获得运行资格。
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        
        # 如果一个请求都没有，就没有任何工作可以交给 ModelRunner。
        if not self.running:
            return [], False
        # 只要当前 running 中还有一条 Sequence 处于 prefill，
        # 就告诉 ModelRunner：这一轮包含 prefill。
        # Day01 这里是教学简化。
        # 后面真正实现时，prefill / decode 会被更精细地分别准备 metadata。
        is_prefill = any(seq.is_prefill for seq in self.running)

        # 返回一个普通 list，避免 ModelRunner 直接修改 Scheduler 内部 deque。
        return list(self.running), is_prefill

    # 优先prefill, 其次decode, 且一个batch里面仅有prefill或者decode
    def schedule(self) -> tuple[list[Sequence], bool]:
        if trace_enabled("TINYINFER_TRACE_SCHED"):
            print(
                "[sched]",
                "waiting=", len(self.waiting),
                "waiting_ids=", [s.seq_id for s in self.waiting],
                "running=", len(self.running),
                "running_ids=", [s.seq_id for s in self.running],
            )

        scheduled: list[Sequence] = []
        token_budget = self.max_num_batched_tokens # 一轮调度中最多支持多少tokens
        # ------------------------------------------------------------
        # Phase A: admit waiting requests for prefill
        # ------------------------------------------------------------
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0] # 所有需要prefill的seq必然位于waiting队列中
            prefill_tokens = seq.num_tokens - seq.num_cached_tokens
            if prefill_tokens > token_budget:
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
            seq.is_prefill = False
            seq.mark_scheduled(1)
            scheduled.append(seq)
            token_budget -= 1
        return scheduled, False


    def postprocess_naive(self, seqs: list[Sequence], token_ids: list[int]):
        """
        ModelRunner 完成本轮 forward 后，把生成结果写回 Sequence。
        seqs[i] 和 token_ids[i] 一一对应：
            seqs[0] -> token_ids[0]
            seqs[1] -> token_ids[1]
            ...
        每条 Sequence：
        1. 追加本轮新生成的 token；
        2. 第一次 forward 后离开 prefill，进入 decode；
        3. 检查 max_tokens / EOS；
        4. 已完成的 Sequence 从 running 中移除。
        """
        finished = []
        for seq, token_id in zip(seqs, token_ids):
            # 把模型本轮生成的新 token 写回 Sequence。
            seq.append_token(token_id)

            # Day01 简化：只要跑过一次，就认为 prefill 已完成，
            # 后面的轮次都进入 autoregressive decode。
            seq.is_prefill = False

            # 检查是否达到 max_tokens 或遇到 EOS。
            if seq.should_stop(self.eos_token_id):
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)

        # 把本轮已经完成的 Sequence 从 running queue 中删除。
        if finished:
            done_ids = {seq.seq_id for seq in finished}
            self.running = deque(
                seq for seq in self.running
                if seq.seq_id not in done_ids
            )

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        finished = []
        for seq, token_id in zip(seqs, token_ids): # 针对这一批次每个前向传播完成的Seq进行操作
            # prefill/decode 本轮都确认了原有 token 对应 KV 已经计算完成
            seq.num_cached_tokens = seq.num_tokens

            # sampled token 加入 sequence；这个新 token 的 KV 要等下一轮 decode
            seq.append_token(token_id)
            seq.num_scheduled_tokens = 0
            seq.is_prefill = False

            if seq.should_stop(self.eos_token_id, self.max_model_len):
                seq.status = SequenceStatus.FINISHED

            if seq.should_stop(self.eos_token_id, self.max_model_len): # 判断停止条件: 出现eos或者达到max_tokens
                seq.status = SequenceStatus.FINISHED
                finished.append(seq) # 放入finished列表

        done_ids = {seq.seq_id for seq in finished} # 已完成的seq编号
        if done_ids:
            self.running = deque(
                seq for seq in self.running if seq.seq_id not in done_ids
            )
            
