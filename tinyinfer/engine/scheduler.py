from collections import deque

from tinyinfer.config import Config
from tinyinfer.engine.sequence import Sequence, SequenceStatus


# 调度器, 本身不负责运行前向传播, 而是选择合适的sequence给ModelRunner进行前向传播

class Scheduler:
    def __init__(self, config: Config):
        # 一轮最多允许多少条 Sequence 同时处于 running。
        # Day01 暂时只真正使用 max_num_seqs；
        # max_num_batched_tokens 会在后续 token-budget / chunked prefill 中使用。
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
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

    def schedule(self) -> tuple[list[Sequence], bool]: # 核心逻辑: 筛选合适的sequence给ModelRuner进行前向传播
        """
        决定本轮让哪些 Sequence 执行。

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
        #
        # Day01 这里是教学简化。
        # 后面真正实现时，prefill / decode 会被更精细地分别准备 metadata。
        is_prefill = any(seq.is_prefill for seq in self.running)

        # 返回一个普通 list，避免 ModelRunner 直接修改 Scheduler 内部 deque。
        return list(self.running), is_prefill

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
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