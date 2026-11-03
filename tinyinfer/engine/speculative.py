from dataclasses import dataclass
import torch


# 从概率分布进行采样, 这里的概率分布可能是p(x), q(x), 或者是max(p(x) - q(x), 0);
# q(x)是draft model对logits进行topk筛选 + softmax之后的归一化概率分布向量
# p(x)是target model对logits进行topk筛选 + softmax之后的
def sample_from_distribution(probs: torch.Tensor) -> int:
    if probs.ndim != 1:
        raise ValueError("probs must be one-dimensional")
    return int(torch.multinomial(probs, num_samples=1).item())




# 对于某一个rejected token进行residual重采样, 即从max(p(x) - q(x), 0)这个概率向量中直接进行采样
def sample_residual_correction(
    target_probs: torch.Tensor, # [vocab]
    draft_probs: torch.Tensor   # [vocab]
) -> int:
    if target_probs.shape != draft_probs.shape or target_probs.ndim != 1:
        raise ValueError("Fatal: shape of target/draft distributions mismatch.")
    
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    mass = residual.sum()

    if not torch.isfinite(mass).item() or mass.item() <= 0.0:
        raise RuntimeError("Fatal: Residual mass is zero or non-finite after rejection.")

    residual = residual / mass
    return int(torch.multinomial(residual, num_samples=1).item())



# 对完整draft token序列进行批量并行验证; target_probs比draft_probs多最后一行, 用于全接受时采样bonus token
# 可以统一处理greedy/非greedy采样, 因为greedy时P/Q矩阵每一行都是one-hot, 可以复用标准SD的拒绝+重采样逻辑
def speculative_verify(
    proposals: list[int],       # [K] 
    draft_probs: torch.Tensor,  # [K, vocab_size]
    target_probs: torch.Tensor, # [K + 1, vocab_size], 每一行代表相应token的p(x), 最后一行代表bonus token
    eos_token_id: int,          # EOS token id
) -> tuple[list[int], int]:     # [最终SD结果(proposal + correction/bonus, 不包含pending), accepted_len]
    K = len(proposals)
    if target_probs.ndim != 2 or draft_probs.ndim != 2:
        raise ValueError("Fatal: target/draft distributions must be 2-D.")
    if draft_probs.shape[0] != K or target_probs.shape[0] != K + 1 or target_probs.shape[1] != draft_probs.shape[1]:
        raise ValueError("Fatal: Expected draft_probs=[K,V] and target_probs=[K+1,V].")

    # K=0时不存在proposal, 直接从target最后一行采样bonus token
    if K == 0:
        return [sample_from_distribution(target_probs[0])], 0

    proposal_ids = torch.tensor(proposals, dtype=torch.long, device=target_probs.device)
    rows = torch.arange(K, dtype=torch.long, device=target_probs.device)
    p_x = target_probs[rows, proposal_ids] # 批量获取每个proposal在Target/Draft分布中的p(x)、q(x)
    q_x = draft_probs[rows, proposal_ids]
    if torch.any(q_x <= 0):
        raise RuntimeError("Fatal: A proposed token has zero probability under the draft distribution.")

    # 并行对K个proposal token进行接受判断, 得到形如[True, Ture, False, True]的mask, 其中第一个False即为拒绝点
    accept_probs = torch.clamp(p_x / q_x, max=1.0)
    accepted_mask = torch.rand_like(accept_probs) < accept_probs
    rejected_rows = torch.nonzero(~accepted_mask, as_tuple=False) # 返回所有True的下标
    reject_idx = int(rejected_rows[0].item()) if rejected_rows.numel() > 0 else K # 找到第一个reject位置; 该位置之后的proposal结果全部无效

    # 只检查第一个reject之前真正被接受的proposal; 若其中出现EOS则立即截断返回
    accepted_proposals = proposals[:reject_idx]
    if eos_token_id in accepted_proposals:
        eos_idx = accepted_proposals.index(eos_token_id)
        return accepted_proposals[:eos_idx + 1], eos_idx + 1

    if reject_idx == K: # 全被接受, 从P矩阵最后一行进行采样获取bonus token
        bonus_token = sample_from_distribution(target_probs[K])
        return proposals + [bonus_token], K

    # reject位置从residual distribution max(p-q, 0)重新采样correction token
    correction_token = sample_residual_correction(target_probs[reject_idx], draft_probs[reject_idx])
    return proposals[:reject_idx] + [correction_token], reject_idx







