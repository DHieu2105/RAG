"""Character and document-structure-aware chunking algorithms."""


STRUCTURE_FIELDS = (
    "part",
    "part_title",
    "chapter",
    "chapter_title",
    "article",
    "article_title",
)


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


def _structure_key(block: dict) -> tuple:
    """Identify the structural container that a block belongs to."""
    return tuple(block.get(field) for field in STRUCTURE_FIELDS)


def _copy_structure(source: dict, target: dict) -> None:
    for field in STRUCTURE_FIELDS:
        value = source.get(field)
        if value is not None:
            target[field] = value


def _clause_numbers(blocks: list[dict]) -> list[int]:
    clauses: list[int] = []
    for block in blocks:
        clause = block.get("clause")
        if clause is not None and clause not in clauses:
            clauses.append(clause)
    return clauses


def structure_chunk(
    blocks: list[dict],
    max_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """Group blocks without crossing part, chapter, article, or size boundaries."""
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

        page_starts = [
            block["page_start"]
            for block in current_blocks
            if block.get("page_start") is not None
        ]
        page_ends = [
            block["page_end"]
            for block in current_blocks
            if block.get("page_end") is not None
        ]
        chunk = {
            "article": current_blocks[0].get("article"),
            "clauses": _clause_numbers(current_blocks),
            "page_start": min(page_starts) if page_starts else None,
            "page_end": max(page_ends) if page_ends else None,
            "text": "\n\n".join(
                block["text"] for block in current_blocks
            ),
        }
        _copy_structure(current_blocks[0], chunk)
        chunks.append(chunk)
        current_blocks = []
        current_length = 0

    for block in blocks:
        text = block["text"]
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
                    "article": block.get("article"),
                    "clauses": _clause_numbers([block]),
                    "page_start": block.get("page_start"),
                    "page_end": block.get("page_end"),
                    "text": text_part,
                }
                _copy_structure(block, chunk)
                chunks.append(chunk)
            continue

        separator_length = 2 if current_blocks else 0
        combined_length = current_length + separator_length + len(text)
        structure_changed = (
            bool(current_blocks)
            and _structure_key(current_blocks[0]) != _structure_key(block)
        )

        if structure_changed or combined_length > max_size:
            save_current_chunk()
            separator_length = 0

        current_blocks.append(block)
        current_length += separator_length + len(text)

    save_current_chunk()
    return chunks

