import torch
from torch import nn

from tinyinfer.layers.activation import SiluAndMul
from tinyinfer.layers.attention import Attention
from tinyinfer.layers.embed_head import LMHead, VocabEmbedding
from tinyinfer.layers.layernorm import RMSNorm
from tinyinfer.layers.linear import (
    LinearBase,
    MergedGateUpLinear,
    QKVParallelLinear,
)
from tinyinfer.layers.rotary_embedding import RotaryEmbedding


# 辅助函数: 获取rope的base角频率
def _get_rope_theta(config):
    if hasattr(config, "rope_theta"): # 存在config.rope_theta, 直接返回
        return config.rope_theta

    rope_scaling = getattr(config, "rope_scaling", None)

    if rope_scaling is not None: # 不存在config.rope_theta, 但是存在config.rope_scaling
        return rope_scaling.get("rope_theta", 10000.0) # 存在config.rope_scaling.rope_theta则直接返回, 否则默认返回10000

    return 10000.0 # 既不存在config.rope_theta也不存在config.rope_scaling, 默认返回10000





# 组装Attention
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx:int, kv_cache=None):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        # 如果config存在head_dim成员变量则直接用head_dim, 否则进入fallback
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        
        self.q_size = self.num_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=bool(getattr(config, "attention_bias", False)),
        )

        # Qwen3 使用 per-head Q/K RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            head_dim = self.head_dim,
            rotary_dim = self.head_dim, # 默认所有head_dim都rot
            base = _get_rope_theta(config)
        )

        self.attn = Attention(
            layer_idx = layer_idx,
            num_heads = self.num_heads,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            scale = self.head_dim ** (-0.5),
            kv_cache = kv_cache,
            block_size=int(getattr(config, "kvcache_block_size", 256))
        )

        self.o_proj = LinearBase(
            self.q_size,
            config.hidden_size,
            bias=bool(getattr(config, "attention_bias", False))
        )

    def set_kv_cache(self, kv_cache) -> None:
        self.attn.set_kv_cache(kv_cache)

    # hidden_states: [num_total_tokens, hidden_size]
    # 整体shape变化: [num_total_tokens, hidden_size] -> [num_total_tokens, hidden_size]
    def forward(self, hidden_states:torch.Tensor, positions:torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv)
    
        q = q.view(-1, self.num_heads, self.head_dim) # -1代表自动推导的维度
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Qwen3 的 q_norm/k_norm 只沿每个 head 的 head_dim 做 RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # positions 直接来自 mixed-batch metadata
        q, k = self.rotary_emb(q, k, positions)
        
        # Attention 统一处理 prefill/decode/mixed; 输出shape=[T, Hq, D]
        out = self.attn(q, k, v) # 内部会根据context内容进行token拆分(为不同seq请求)
        out = out.reshape(-1, self.q_size) # shape = [T, Hq * D]
        return self.o_proj(out) # shape = [T, hidden_size]


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedGateUpLinear(
            config.hidden_size,
            config.intermediate_size,
            bias = False
        )
        self.act = SiluAndMul()
        self.down_proj = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            bias = False
        )

    # 完整的SwiGLU, shape变化: [num_flat_tokens, hidden_size] -> [num_flat_tokens, hidden_size]
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.act(x)
        return self.down_proj(x)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx:int, kv_cache = None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_idx, kv_cache)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def set_kv_cache(self, kv_cache) -> None:
        self.self_attn.set_kv_cache(kv_cache)

    # 关键前向传播过程, 包含两个pre-norm, 分别针对attention计算和线性层; 各自都包含residual跳连
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# 完整模型Backbone, embedding layer + multi-layer transformer, 但不包含最后的LM Head
class Qwen3Model(nn.Module):
    def __init__(self, config, kv_cache=None):
        super().__init__()
        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i, kv_cache) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def set_kv_cache(self, kv_cache) -> None:
        for layer in self.layers:
            layer.set_kv_cache(kv_cache)
    
    # 完整的前向传播流程
    def forward(self, input_ids:torch.Tensor, positions:torch.Tensor, output_hidden_states: bool = False):
        hidden_states = self.embed_tokens(input_ids) # [num_flat_tokens] -> [num_flat_tokens, hidden_dim], 把token_id变为token向量
        
        if not output_hidden_states: # 正常推理路径：不保存任何中间结果
            for layer in self.layers:
                hidden_states = layer(hidden_states, positions)
            hidden_states = self.norm(hidden_states)
            return hidden_states, None

        # debug / 等价性测试路径
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
            all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        # 为了和 HF hidden_states[-1] 对齐，用 final RMSNorm 输出替换最后一个 decoder layer raw output
        all_hidden_states[-1] = hidden_states

        return hidden_states, all_hidden_states


# Backbone + 最后的LM Head
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config, kv_cache=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, kv_cache)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        
        # 若 checkpoint 使用 tied embeddings(即weight tying, 权重绑定), 让两个 module 共享同一个 Parameter
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_kv_cache(self, kv_cache) -> None:
        self.model.set_kv_cache(kv_cache)

    # forward等价于backbone中的forward, 和最终的compute_logits解耦
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, output_hidden_states: bool = False) -> torch.Tensor:
        # 只返回 hidden states；mixed batch 中并非所有 token 都需要 logits
        return self.model(input_ids, positions, output_hidden_states)

    # compute_logits 产出[num_flat_tokens, vocab_size], 仍然不是最终的next_token id list; 还差一个sampler
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # forward + compute_logits + sampler才会生成next token id






