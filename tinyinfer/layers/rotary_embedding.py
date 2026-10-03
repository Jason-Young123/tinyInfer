import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Qwen/Llama half-split rotary layout 的教学/reference 实现。
    输入：
        q: [num_tokens, num_q_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        positions: [num_tokens]

    注意 mixed batch 中 positions 不要求单调，因为每个 flat token 都带自己的
    absolute sequence position。
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int | None = None,
        base: float = 10000.0,
    ):
        super().__init__()

        if rotary_dim is None:
            rotary_dim = head_dim
        if rotary_dim <= 0:
            raise ValueError("rotary_dim must be positive")
        if rotary_dim > head_dim:
            raise ValueError("rotary_dim cannot exceed head_dim")
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )

        # inv_freq 是模型状态且为tensor, 但不是 checkpoint parameter; 又希望能够使用.cuda/.to等管理函数, 因此要注册为buffer而非
        # 简单的成员变量; 简单的成员变量在model.to()的时候数据类型、device都不会跟随着变化
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def _apply(self, fn, recurse=True):
        """
        覆盖 nn.Module._apply。
        目的:
            model.to(dtype=torch.bfloat16)会递归调用 _apply, buffer也会dtype转换。
        但是:
            inv_freq需要保持FP32
        所以:
            允许device变化但禁止dtype变化。
        """
        # 保存原始FP32 inv_freq
        inv_freq_fp32 = self.inv_freq

        # 正常执行parameter转换和其他buffer转换
        super()._apply(fn, recurse)

        # 恢复inv_freq:只跟随device
        self.inv_freq = inv_freq_fp32.to(device=self.inv_freq.device)

        return self





    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """对 x 的前 rotary_dim 维应用 half-split RoPE。"""
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        half = self.rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]

        # cos/sin: [T, half] -> [T, 1, half]，广播到所有 heads。
        cos = cos.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        sin = sin.unsqueeze(1).to(dtype=x.dtype, device=x.device)

        # 等价于 HF rotate_half 的 half-split layout：
        # [x1, x2] -> [-x2, x1]。
        first = x1 * cos - x2 * sin
        second = x2 * cos + x1 * sin
        rotated = torch.cat((first, second), dim=-1)

        if x_pass.shape[-1] == 0:
            return rotated
        return torch.cat((rotated, x_pass), dim=-1)




    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("q/k must be [num_tokens, num_heads, head_dim]")
        if positions.ndim != 1:
            raise ValueError("positions must be a 1-D tensor")
        if q.shape[0] != k.shape[0] or q.shape[0] != positions.numel():
            raise ValueError("q, k and positions must have the same token count")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("q/k head_dim mismatch")

        positions_fp32 = positions.to(device=q.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=q.device, dtype=torch.float32)

        # [T] outer [rotary_dim/2] -> [T, rotary_dim/2]
        freqs = torch.outer(positions_fp32, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k
