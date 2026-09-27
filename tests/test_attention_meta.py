from tinyinfer.engine.model_runner import token_to_slot


def test_token_to_slot():
    class S:
        block_size = 4
        block_table = [7, 2, 11]
    
    
    assert token_to_slot(S(), 0) == 28
    assert token_to_slot(S(), 6) == 10
    assert token_to_slot(S(), 8) == 44
