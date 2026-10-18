"""Build readable LLM context from retrieved document chunks."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with their source information."""
    contexts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        article = chunk.get("article")
        clauses = chunk.get("clauses")
        if clauses is None:
            clause = chunk.get("clause")
            clauses = [clause] if clause is not None else []
        page = chunk.get("page_start")
        text = chunk["text"]
        clauses_text = ", ".join(str(clause) for clause in clauses)

        context = (
            f"[Tài liệu {index}]\n"
            f"Điều: {article}\n"
            f"Khoản: {clauses_text}\n"
            f"Trang PDF: {page}\n"
            "Nội dung:\n"
            f"{text}"
        )
        contexts.append(context)

    return "\n\n".join(contexts)
