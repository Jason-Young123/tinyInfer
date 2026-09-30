from transformers import AutoTokenizer
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ToyModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


# 整体调用链条: LLM -> LLMEngine -> 创建seqs列表 -> Scheduler -> ModelRunner

class LLMEngine:
    def __init__(self, config: Config | None = None, device: str = "cuda"):
        self.config = config or Config()
        self.config.validate_runtime_fields()
        self.config.load_hf_config()

        Sequence.block_size = self.config.kvcache_block_size

        # ModelRunner 先初始化，因为它会根据真实模型显存占用计算config.num_kvcache_blocks;
        # 随后 Scheduler/BlockManager 才能使用该值。
        self.model_runner = ModelRunner(self.config, device=device)
        self.scheduler = Scheduler(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        self.config.eos_token_id = (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1)
        self.scheduler.eos_token_id = self.config.eos_token_id


    def add_request(self, token_ids: list[int], params: SamplingParams): # 简化后的创建请求函数, 用一个token list代表
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        return seq.seq_id
    
    def step(self): # 最重要函数之一
        output = self.scheduler.schedule()
        if not output.items:
            return []
        sampled_tokens = self.model_runner.run(output)
        self.scheduler.postprocess(output, sampled_tokens)
        return output.items

    # 针对所有seq请求生成/decode完整的token id序列
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
            items = self.step()
            if not items and not self.scheduler.is_finished():
                raise RuntimeError("scheduler made no progress while requests remain")

        seqs.sort(key=lambda x: x.seq_id)
        return [
            {
                "token_ids": seq.token_ids[seq.num_prompt_tokens:], # decode阶段生成的token id
                "all_token_ids": list(seq.token_ids), # 所有的token_id
                "num_cached_tokens": seq.num_cached_tokens
            }
            for seq in seqs
        ]


    # 上层最终调用的接口, 输入输出均为str列表
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        # step1: str -> token_id
        prompt_token_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]

        # step2: 运行直到每个seq请求都推理完毕, 得到完整的token_ids列表
        outputs = self.generate_token_ids(prompt_token_ids, sampling_params)

        # step3: 将输出的token_ids 解码为字符串
        for out in outputs:
            out["text"] = self.tokenizer.decode(out["token_ids"], skip_special_tokens=True) # 动态为dict新建"text"键

        return outputs





