from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config

# 实际上创建LLMEngine类
myllm = LLM(Config(max_num_seqs = 2)) # 最多同时允许两条请求并行推进


outputs = myllm.generate_token_ids(
    [[10, 20], [100]],
    SamplingParams(temperature=1.0, max_tokens=3, ignore_eos=True), # 每条请求最多允许产生3个token
)

for i, out in enumerate(outputs):
    print(i, out)
