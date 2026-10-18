"""Character-based chunking algorithms implemented from scratch."""


def fixed_size_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split text into fixed-size chunks with a character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += step

    return chunks


def structure_chunk(
    blocks: list[dict],
    max_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """Group clause blocks without crossing article or size boundaries."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")
    if overlap < 0 or overlap >= max_size:
        raise ValueError("overlap must be between 0 and max_size - 1")

    chunks: list[dict] = []
    current_blocks: list[dict] = []
    current_length = 0

    def save_current_chunk() -> None:
        nonlocal current_blocks, current_length

        if not current_blocks:
            return

        chunk = {
            "article": current_blocks[0]["article"],
            "clauses": [
                block["clause"]
                for block in current_blocks
                if block.get("clause") is not None
            ],
            "page_start": current_blocks[0]["page_start"],
            "page_end": current_blocks[-1]["page_end"],
            "text": "\n\n".join(
                block["text"] for block in current_blocks
            ),
        }
        if "article_title" in current_blocks[0]:
            chunk["article_title"] = current_blocks[0]["article_title"]
        if "chapter_title" in current_blocks[0]:
            chunk["chapter_title"] = current_blocks[0]["chapter_title"]
        chunks.append(chunk)
        current_blocks = []
        current_length = 0

    for block in blocks:
        text = block["text"]
        article = block["article"]

        if not text:
            continue

        if len(text) > max_size:
            save_current_chunk()

            for text_part in fixed_size_chunk(
                text,
                chunk_size=max_size,
                overlap=overlap,
            ):
                chunk = {
                    "article": article,
                    "clauses": (
                        [block["clause"]]
                        if block.get("clause") is not None
                        else []
                    ),
                    "page_start": block["page_start"],
                    "page_end": block["page_end"],
                    "text": text_part,
                }
                if "article_title" in block:
                    chunk["article_title"] = block["article_title"]
                if "chapter_title" in block:
                    chunk["chapter_title"] = block["chapter_title"]
                chunks.append(chunk)
            continue

        separator_length = 2 if current_blocks else 0
        combined_length = current_length + separator_length + len(text)
        article_changed = (
            bool(current_blocks)
            and current_blocks[0]["article"] != article
        )

        if article_changed or combined_length > max_size:
            save_current_chunk()
            separator_length = 0

        current_blocks.append(block)
        current_length += separator_length + len(text)

    save_current_chunk()
    return chunks

