"""Build readable LLM context from retrieved document chunks."""


def _metadata_value(identifier: object, title: object) -> str | None:
    if identifier is None and not title:
        return None
    if identifier is None:
        return str(title)
    if not title:
        return str(identifier)
    return f"{identifier} — {title}"


def _page_value(page_start: object, page_end: object) -> str | None:
    if page_start is None and page_end is None:
        return None
    if page_end is None or page_end == page_start:
        return str(page_start)
    if page_start is None:
        return str(page_end)
    return f"{page_start}-{page_end}"


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with only their available source metadata."""
    contexts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        clauses = chunk.get("clauses")
        if clauses is None:
            clause = chunk.get("clause")
            clauses = [clause] if clause is not None else []

        metadata_lines = [f"[Tài liệu {index}]"]
        structural_metadata = (
            (
                "Phần",
                _metadata_value(chunk.get("part"), chunk.get("part_title")),
            ),
            (
                "Chương",
                _metadata_value(
                    chunk.get("chapter"), chunk.get("chapter_title")
                ),
            ),
            (
                "Điều",
                _metadata_value(
                    chunk.get("article"), chunk.get("article_title")
                ),
            ),
        )
        for label, value in structural_metadata:
            if value is not None:
                metadata_lines.append(f"{label}: {value}")

        if clauses:
            clauses_text = ", ".join(str(clause) for clause in clauses)
            metadata_lines.append(f"Khoản: {clauses_text}")

        page = _page_value(chunk.get("page_start"), chunk.get("page_end"))
        if page is not None:
            metadata_lines.append(f"Trang PDF: {page}")

        metadata_lines.extend(("Nội dung:", chunk["text"]))
        contexts.append("\n".join(metadata_lines))

    return "\n\n".join(contexts)
