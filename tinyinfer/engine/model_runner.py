
class ToyModelRunner:
    """Control-plane-only runner used in Day01/02.

    It does not run a neural network. For every sequence it simply emits
    `(last_token + 1) % 1000` so that the engine loop can be tested.
    """

    def run(self, sequences, is_prefill: bool): # 当前阶段不论是不是prefill都默认在末尾添加 id+1
        next_tokens = []
        for seq in sequences:
            next_tokens.append((seq.last_token + 1) % 1000)
        return next_tokens




