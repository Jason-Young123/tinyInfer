import torch
from torch import nn


# 对若干独立的 logits 向量分别进行采样, 输出采样结果(和分布向量)
# 支持同一个 mixed batch 中每条请求独立 greedy/temperature; 支持输出/不输出概率分布
# 如果需要输出概率分布, 整体计算流程为: softmax 求每一行概率分布 + multinomial 采样
# 如果不需要输出概率分布, 整体计算流程为: 直接对每一行进行 Gumbel 采样
class Sampler(nn.Module): # shape变化: [sampled_seq, vocab_size] -> [sampled_seq], 即每个需要进行sample的seq请求最终采样到哪一个token
    def forward(
        self,
        logits: torch.Tensor,                       # [N, V]
        temperatures: torch.Tensor,                 # [N]
        topks: torch.Tensor,                        # [N]
        greedy_mask: torch.Tensor | None = None,    # [N]
        isDistributionOut: bool = False             # bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:  # [N], [N, V]
        if logits.ndim != 2:
            raise ValueError("logits must be [num_samples, vocab_size]")
        n = logits.shape[0]
        vocab_size = logits.shape[1]
        if temperatures.shape != (n,):
            raise ValueError("temperatures must be [num_samples]")
        if topks.shape != (n,):
            raise ValueError("topks must be [num_samples]")

        if greedy_mask is None: # 默认不启用greedy sampling
            greedy_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
        if greedy_mask.shape != (n,):
            raise ValueError("greedy_mask must be [num_samples]")

        result = torch.empty(n, dtype=torch.long, device=logits.device)
        distributions = torch.zeros((n, vocab_size), dtype=torch.float32, device=logits.device) if isDistributionOut else None

        # Greedy 行直接 argmax
        if greedy_mask.any():
            result[greedy_mask] = logits[greedy_mask].argmax(dim=-1) # 仅采样greedy_mask为True的行
            if isDistributionOut:
                rows = greedy_mask.nonzero(as_tuple=False).squeeze(-1)
                distributions[rows, result[rows]] = 1.0 # 把相应行的指定列设为1.0

        # 非 greedy 行再做 sampling
        sample_mask = ~greedy_mask
        if sample_mask.any():
            temps = temperatures[sample_mask]
            ks = topks[sample_mask].long()
            
            if torch.any(temps <= 0):
                raise ValueError("non-greedy temperatures must be > 0")
            if torch.any(ks <= 0) or torch.any(ks > vocab_size):
                raise ValueError("top-k must satisfy 1 <= top-k <= vocab_size")

            scaled = (logits[sample_mask] / temps.unsqueeze(-1)).float()    # [N,V]
            max_k = ks.max().item()
            topk_values = torch.topk(scaled, k=max_k, dim=-1).values        # [N, max_k] ; 每一行返回max_k个降序排列的logits值
            threshold = topk_values.gather(1, (ks - 1).unsqueeze(-1))       # [N,1]; 每一行取自己的第 k 大元素作为 threshold
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))  # [N, V]

            if isDistributionOut: # 要输出distribution向量, 直接用softmax + multinomial采样
                probs = torch.softmax(scaled, dim=-1)
                result[sample_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                distributions[sample_mask] = probs
            else: # 无需输出distribution向量, 直接用基于Gumbel-max的softmax-free采样
                u = torch.rand_like(scaled).clamp_min_(1e-10) # U ~ Uniform(0, 1); 因为rand_like生成的范围是[0, 1)而非(0, 1), 故要用clamp约束下界
                gumbel = -torch.log(-torch.log(u)) # G ~ Gumbel(0, 1)
                result[sample_mask] = (scaled + gumbel).argmax(dim=-1)

        return result, distributions




