from dataclasses import dataclass
import torch

from transformers import AutoTokenizer
from tinyinfer.config import Config
from tinyinfer.engine.model_runner import ModelRunner
from tinyinfer.engine.scheduler import Scheduler
from tinyinfer.engine.sequence import Sequence
from tinyinfer.sampling_params import SamplingParams


@dataclass(slots=True)
class StreamUpdate:
    seq_id: int
    token_id: int
    completion_token_ids: list[int]
    finished: bool
    statistics: dict | None


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
 
    def build_chat_prompt(self, text:str, system_prompt:str|None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        if self.tokenizer.chat_template:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        return text

    # 把 UI 字符串转换成真正 chat-template prompt, 再进入原 Scheduler
    def add_text_request(
        self,
        text: str,
        params: SamplingParams,
        system_prompt: str | None = None,
    ) -> int:
        prompt = self.build_chat_prompt(text, system_prompt)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.add_request(token_ids, params)


    def step_with_updates(self) -> tuple[list, list[StreamUpdate]]:
        # 对内: 走通schedule -> run -> postprocess的完整一轮调度流程
        output = self.scheduler.schedule() # 从waiting池子中挑选本轮进行推理的请求
        if not output.items:
            return [], []
        sampled_tokens = self.model_runner.run(output) # 输出为dict of {seq_id: token_id}, 表示所有需要采样的请求所产生的下一个token id是什么
        self.scheduler.postprocess(output, sampled_tokens) # 将产生的下一个token拼接到对应的seq请求中, 并注册prefix cache; 管理waiting/running list

        # 对外: 把当前一轮真正生成token的请求信息打包送给前端, 进行网页端聊天框刷新
        updates = []
        for item in output.items:
            token_id = sampled_tokens.get(item.seq.seq_id) # 得到产生next token请求的next token id
            if token_id is None: # 如果这一批的某个seq没有生成下一个token则直接跳过, 无需进行前端网页刷新
                continue
            seq = item.seq
            updates.append(
                StreamUpdate(
                    seq_id=seq.seq_id,
                    token_id=token_id,
                    completion_token_ids=list(
                        seq.token_ids[seq.num_prompt_tokens:]
                    ),
                    finished=seq.is_finished,
                    statistics=seq.statistics() if seq.is_finished else None,
                )
            )

        return output.items, updates # 前者是内部信息(这一轮调度了哪些请求, 不论是否生成next token); 后者是对外信息(这一轮哪些请求生成了next token从而需要刷新聊天框)

    def step(self): # 对内的前向传播调度函数, 不对外传递信息
        items, _ = self.step_with_updates()
        return items

    
    # 资源快照只读取 CPU metadata，不做 CUDA synchronize。
    def runtime_resource_snapshot(self) -> dict:
        device = self.model_runner.device
        if device.type != "cuda": # cpu端直接忽略
            return {
                "total_bytes": 0,
                "unused_bytes": 0,
                "reserved_bytes": 0,
                "allocated_kv_bytes": 0,
                "available_kv_bytes": 0,
                "block_bytes": 0,
                "num_blocks": 0,
                "active_block_ids": [],
            }

        profile = self.model_runner.resource_profile
        total_bytes = profile.total_bytes
        unused_bytes = profile.unused_bytes
        reserved_bytes = profile.reserved_bytes
        block_bytes = profile.block_bytes
        blocks = self.scheduler.block_manager.blocks
        active_block_ids = [
            block.block_id for block in blocks if block.ref_count > 0
        ]
        allocated_kv_bytes = len(active_block_ids) * block_bytes
        available_kv_bytes = max(0, total_bytes - unused_bytes - reserved_bytes - allocated_kv_bytes)

        return {
            "total_bytes": total_bytes,
            "unused_bytes": unused_bytes,
            "reserved_bytes": reserved_bytes,
            "allocated_kv_bytes": int(allocated_kv_bytes),
            "available_kv_bytes": int(available_kv_bytes),
            "block_bytes": int(block_bytes),
            "num_blocks": len(blocks),
            "active_block_ids": active_block_ids,
        }




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
        for token_ids, p in zip(prompts, params): # seq达到一刻
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
                "token_ids": seq.token_ids[seq.num_prompt_tokens:], # decode阶段生成的token id, 不包含prompt
                "all_token_ids": list(seq.token_ids), # 所有的token_id
                "num_cached_tokens": seq.num_cached_tokens, # prefix cache命中的token数目
                "statistics": seq.statistics(),
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





