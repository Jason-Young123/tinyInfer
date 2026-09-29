import torch

from tinyinfer.layers.embed_head import LMHead, VocabEmbedding


def test_embedding_and_lm_head_shapes():
    vocab_size = 128
    hidden_size = 32

    embedding = VocabEmbedding(vocab_size, hidden_size)
    lm_head = LMHead(hidden_size, vocab_size, bias=False)

    input_ids = torch.tensor([1, 5, 9, 11], dtype=torch.long)
    hidden = embedding(input_ids)
    logits = lm_head(hidden)

    assert hidden.shape == (4, hidden_size)
    assert logits.shape == (4, vocab_size)
