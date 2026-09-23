from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ToyModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


# 整体调用链条: LLM -> LLMEngine -> 创建seqs列表 -> Scheduler -> ModelRunner

class LLMEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        Sequence.block_size = self.config.kvcache_block_size

        self.scheduler = Scheduler(self.config)
        self.model_runner = ToyModelRunner()

    def add_request(self, token_ids: list[int], params: SamplingParams): # 简化后的创建请求函数, 用一个token list代表
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # 返回这一轮可以一起前向推理的seq列表, 以及其中是否包含prefill请求
        if not seqs:
            return []
        next_tokens = self.model_runner.run(seqs, is_prefill) # 调用ModelRunner跑一步
        self.scheduler.postprocess(seqs, next_tokens) # ModelRunner跑完之后交由Scheduler进行后处理; 这一步会修改seqs中的每一个seq
        return seqs

    def generate_token_ids(
        self,
        prompts: list[list[int]], # 每个请求本身是一个list,因此这里是list的list
        sampling_params: SamplingParams | list[SamplingParams], # 如果只有一个sampling_params说明所有请求共用, 需要进行广播
    ):
        # basic argument check
        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = sampling_params
        if len(params) != len(prompts):
            raise ValueError("prompts and sampling params length mismatch")

        # 创建Sequence列表并加入到Scheduler中由其进行调度
        seqs = []
        for token_ids, p in zip(prompts, params):
            seq = Sequence(token_ids, p)
            seqs.append(seq)
            self.scheduler.add(seq)

        # 直到Scheduler调度完成, 所有请求全部推理完毕; 实际上Scheduler会调用ModelRunner进行真正的推理
        while not self.scheduler.is_finished():
            self.step()

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:],
                "all_token_ids": seq.token_ids,
            }
            for seq in seqs
        ]
