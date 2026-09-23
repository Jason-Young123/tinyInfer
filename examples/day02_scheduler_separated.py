from tinyinfer import LLM, SamplingParams
from tinyinfer.config import Config


prompts = [
    [1] * 800,
    [2] * 20,
    [3] * 300,
    [4] * 40,
    [5] * 500,
]

params = [
    SamplingParams(temperature=1.0, max_tokens=n, ignore_eos=True)
    for n in [30, 3, 12, 5, 20]
]

llm = LLM(
    Config(
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    )
)

out = llm.generate_token_ids(prompts, params)

for i, x in enumerate(out):
    print(i, len(x["token_ids"]))




