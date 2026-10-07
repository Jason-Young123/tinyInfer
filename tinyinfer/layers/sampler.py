import torch
from torch import nn

# 支持同一个 mixed batch 中每条请求独立 greedy/temperature
class Sampler(nn.Module): # shape变化: [sampled_seq, vocab_size] -> [sampled_seq], 即每个需要进行sample的seq请求最终采样到哪一个token
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must be [num_samples, vocab_size]")
        if temperatures.shape != (logits.shape[0],):
            raise ValueError("temperatures must be [num_samples]")

        n = logits.shape[0] # sampled_seq总数
        if greedy_mask is None: # 默认不启用greedy sampling
            greedy_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
        if greedy_mask.shape != (n,):
            raise ValueError("greedy_mask must be [num_samples]")

        result = torch.empty(n, dtype=torch.long, device=logits.device)

        # Greedy 行直接 argmax
        if greedy_mask.any():
            result[greedy_mask] = logits[greedy_mask].argmax(dim=-1) # 仅采样greedy_mask为True的行

        # 非 greedy 行再做 sampling
        sample_mask = ~greedy_mask
        if sample_mask.any():
            temps = temperatures[sample_mask]
            if torch.any(temps <= 0):
                raise ValueError("non-greedy temperatures must be > 0")

            # 传统的softmax + multinomial采样
            #scaled = logits[sample_mask] / temps.unsqueeze(-1)
            #probs = torch.softmax(scaled.float(), dim=-1)
            #result[sample_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # 基于Gumbel-max的softmax-free采样
            scaled = (logits[sample_mask] / temps.unsqueeze(-1)).float()
            u = torch.rand_like(scaled).clamp_min_(1e-10) # U ~ Uniform(0, 1); 因为rand_like生成的范围是[0, 1)而非(0, 1), 故要用clamp约束下界
            gumbel = -torch.log(-torch.log(u)) # G ~ Gumbel(0, 1)
            result[sample_mask] = (scaled + gumbel).argmax(dim=-1) # Gumbel-Max sampling

        return result




