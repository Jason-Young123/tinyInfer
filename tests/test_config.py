from tinyinfer.config import Config

def test_config_model():
    cfg = Config(model = "/home/jason/huggingface/Qwen3-0.6B") # 本地模型要用绝对路径
    cfg.load_hf_config()

    print(cfg.hf_config.hidden_size)
    print(cfg.hf_config.num_hidden_layers)
    print(cfg.hf_config.num_attention_heads)
    print(cfg.hf_config.num_key_value_heads)
    print(cfg.hf_config.head_dim)


