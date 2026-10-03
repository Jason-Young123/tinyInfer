# 用于将问题渲染为chat template

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise technical assistant. "
    "Answer accurately using precise technical terminology."
)


def build_prompt(
    question: str,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )




