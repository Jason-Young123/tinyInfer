__version__ = "0.1.0"

from tinyinfer.llm import LLM
from tinyinfer.sampling_params import SamplingParams


# 希望暴露的接口
__all__ = ["SamplingParams", "LLM"]
