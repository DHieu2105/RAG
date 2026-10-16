DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150

_BREAK_SEPARATORS = ("\n\n", ". ", "; ", ": ", "\n", " ")


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    hard_end = min(start + chunk_size, len(text))
    if hard_end == len(text):
        return hard_end

    minimum_end = start + chunk_size // 2
    break_positions = [
        text.rfind(separator, minimum_end, hard_end)
        for separator in _BREAK_SEPARATORS
    ]
    best_break = max(break_positions)
    return best_break + 1 if best_break >= minimum_end else hard_end


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text at natural boundaries with a character-based overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = _find_chunk_end(text, start, chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - chunk_overlap, start + 1)
        next_space = text.find(" ", next_start, min(next_start + 40, end))
        start = next_space + 1 if next_space != -1 else next_start

    return chunks
