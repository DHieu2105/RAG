"""Simple whitespace tokenizer implemented from scratch."""


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it on consecutive whitespace."""
    text = text.lower()
    tokens = text.split()
    return tokens

