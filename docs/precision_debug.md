结论：当前精度问题不止 RoPE，但主因已经确定。仓库代码没有被修改，所有实验脚本和输出都在 /tmp/tinyinfer_precision。

  问题定位

  1. RoPE：确定存在 BF16 精度错误

  Qwen3-0.6B 配置是 rope_theta=1000000、torch_dtype=bfloat16。官方配置

  官方实现保持 inv_freq 为 FP32，计算频率、cos/sin 后才转成输入 dtype。Transformers v4.51.0 实现

  当前 tinyInfer 在 tinyinfer/layers/rotary_embedding.py:46 注册了 FP32 inv_freq，但测试中的：

  tiny_model.to(device, dtype=torch.bfloat16)

  会把这个 buffer 一起转成 BF16。随后 tinyinfer/layers/rotary_embedding.py:92 只移动 device，没有恢复 FP32。

  实验结果：

  - BF16、原始 RoPE：layer 0 q 最大误差约 0.039，k 最大误差约 0.094
  - position 40959 时，q/k 最大误差达到约 5.6/6.3
  - 重新用 FP32 生成 inv_freq 后，RoPE q/k 误差为 0

  这不是 half-split RoPE 公式错误。当前 half-split 公式与 HF 的 rotate_half 数学等价；错误是 inv_freq 在模
  型 .to(dtype=bf16) 时被提前量化。

  建议修改 tinyinfer/layers/rotary_embedding.py：

  def _apply(self, fn, recurse=True):
      inv_freq = self.inv_freq
      super()._apply(fn, recurse)
      self.inv_freq = inv_freq.to(device=self.inv_freq.device)
      return self

  并将 [第 92 行] 改成：

  inv_freq = self.inv_freq.to(
      device=q.device,
      dtype=torch.float32,
  )

  频率和三角函数计算最好放在禁用 autocast 的上下文中，和 HF 保持一致。仅仅在 forward 中 .float() 不够，因为 buffer 之
  前已经被 BF16 量化，信息无法恢复。

  2. QKV packed projection：确定存在 FP32 bitwise 差异

  官方 Qwen3 使用三个独立的 q_proj/k_proj/v_proj。官方实现

  tinyInfer 在 tinyinfer/layers/linear.py:26 把 QKV 合并成一次大矩阵乘法。数学上等价，但 CPU/GPU GEMM 的累加顺序不
  同：

  - layer 0 q：误差 0
  - layer 0 k：最大误差约 4.8e-7
  - k RMSNorm 后：最大误差约 6.1e-5
  - 28 层后 FP32 hidden 最大误差达到约 9e-5

  临时把 QKV 拆成三个独立 F.linear 后，FP32 prefill/decode 的所有 hidden state 和 logits 都逐元素相等。

  建议在 QKVParallelLinear.forward() 中分别计算 q/k/v，再 torch.cat。权重仍可保持 packed 布局，loader 不需要改。

  3. MLP：没有发现逻辑或 dtype 错误

  官方 MLP 是：

  down_proj(silu(gate_proj(x)) * up_proj(x))

  tinyInfer 的 tinyinfer/layers/activation.py 和 tinyinfer/models/qwen3.py 与此一致。官方实现

  在修正 QKV 后：

  - MLP gate/up/SiLU/down 输出均逐元素一致
  - residual、post-attention RMSNorm 也逐元素一致
  - MergedGateUpLinear 的 packed 计算在本机 CPU 上没有独立产生可见误差

  因此当前主要 MLP 误差是上游 QKV 误差传播造成的，不是 SiLU、MLP 顺序或 RMSNorm 实现错误。为了跨设备严格复刻 HF，也可
  以把 tinyinfer/layers/linear.py:49 的 Gate/Up 拆成两个 F.linear，但本机实验没有证明它是当前必需修改点。

  4. Attention：取决于你要对齐 HF 的哪个后端

  tinyInfer 当前在 tinyinfer/layers/attention.py:165 使用 scaled_dot_product_attention。

  本机 HuggingFace 模型的 config._attn_implementation 是 sdpa。在：

  - RoPE 修正后
  - QKV 拆分后

  tinyInfer 的 SDPA attention 与 HF 默认 SDPA 路径在 prefill/decode 中都逐元素一致。

  但是，HF 官方 eager attention 使用：

  matmul
  -> 加 causal mask
  -> FP32 softmax
  -> 转回 q.dtype
  -> matmul

  官方 eager attention 实现

  当 HF 强制使用 attn_implementation="eager"，而 tinyInfer 继续使用 SDPA 时，BF16 layer 0 attention 最大误差约
  0.027，完整模型 hidden 最大误差约 1.5。将 tinyInfer 也改成 eager 后，误差恢复为 0。

  因此：

  - 对齐 HF 默认 sdpa：attention.py 不需要改数学逻辑，但测试时应明确指定 HF 使用 sdpa
  - 对齐 HF eager/reference arithmetic：需要把 _attend_one() 中的 SDPA 替换为显式 matmul + FP32 softmax + matmul

  端到端实验结果

   路径                                   Prefill hidden 最大误差    Decode hidden 最大误差    Logits 最大误差
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━
   原始 FP32 packed                                       3.81e-5                   3.05e-5            3.00e-5
  ─────────────────────────────────────  ─────────────────────────  ────────────────────────  ─────────────────
   FP32，拆分 QKV                                               0                         0                  0
  ─────────────────────────────────────  ─────────────────────────  ────────────────────────  ─────────────────
   原始 BF16 RoPE                                            0.75                      0.75               0.56
  ─────────────────────────────────────  ─────────────────────────  ────────────────────────  ─────────────────
   BF16，恢复 FP32 inv_freq + 拆分 QKV                          0                         0                  0
  ─────────────────────────────────────  ─────────────────────────  ────────────────────────  ─────────────────
   BF16，HF eager vs tiny SDPA                                1.5                       1.5               0.84
  ─────────────────────────────────────  ─────────────────────────  ────────────────────────  ─────────────────
   BF16，双方都 eager                                           0                         0                  0

  测试代码本身还有三个问题

  - tests/test_hf_equivalence.py:44 在无 CUDA 时强制使用 FP32，因此不会暴露 RoPE 的 BF16 问题。
  - tests/test_hf_equivalence.py:119 到 138 行的逐层 hidden-state 检查被三引号注释掉了。
  - tests/test_hf_equivalence.py:199 调用了 hf_layer0.self_attn.attn，当前 Transformers 5.17 的 Qwen3Attention 没有这
    个成员，测试会在 attention 比较前直接失败。应改用官方 eager_attention_forward，或直接比较模型 forward 输出。

  最终建议的修改优先级是：

  1. rotary_embedding.py：保留 FP32 inv_freq，并在 forward 中强制 FP32 频率计算。
  2. linear.py：将 QKVParallelLinear 改为三个独立 F.linear，以消除 FP32 packed GEMM 差异。
  3. 根据目标后端决定是否修改 attention.py：默认 SDPA 不必改；要与 HF eager 算术完全一致则改成显式 FP32 softmax。
  4. 补全测试中的逐层 hidden、最终 logits、prefill/decode 对比，并明确 HF 的 attention backend。

  本机没有 CUDA，因此 BF16 实验是在 CPU 上完成的，已经验证了 dtype 转换和模型算术路径，但 CUDA SDPA/FlashAttention
  kernel 的最终 bitwise 行为仍需在目标 GPU 上复跑。