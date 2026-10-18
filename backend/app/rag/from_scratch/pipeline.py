"""End-to-end indexing pipeline built from the from-scratch components."""

import sys

from app.rag.loader import PDF_PATH, load_pdf

from .chunker import structure_chunk
from .context_builder import build_context
from .embeddings import TfidfVectorizer
from .index_store import INDEX_PATH, load_index, save_index
from .llm import generate_answer
from .preprocessing import preprocess_pages
from .retriever import top_k_search


def build_index():
    """Build an in-memory TF-IDF index from the student handbook PDF."""
    pages = load_pdf(PDF_PATH)
    blocks = preprocess_pages(pages)
    chunks = structure_chunk(blocks)
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    document_vectors = vectorizer.transform(texts)

    return chunks, vectorizer, document_vectors


def ask(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
) -> list[dict]:
    """Retrieve the five chunks most relevant to a question."""
    query_vector = vectorizer.transform([question])[0]
    return top_k_search(
        query_vector,
        document_vectors,
        chunks,
        top_k=5,
    )


def chat(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
) -> str:
    """Retrieve relevant chunks and use Ollama to answer the question."""
    result = chat_with_sources(
        question,
        chunks,
        vectorizer,
        document_vectors,
    )
    return result["answer"]


def chat_with_sources(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    history: list[dict[str, str]] | None = None,
) -> dict:
    """Return an answer together with the retrieved source metadata."""
    results = ask(
        question,
        chunks,
        vectorizer,
        document_vectors,
    )
    if not results:
        return {"answer": "Không tìm thấy thông tin liên quan trong sổ tay.", "sources": []}

    context = build_context(results)
    answer = (
        generate_answer(question, context, history)
        if history is not None
        else generate_answer(question, context)
    )
    sources = [
        {
            "article": chunk.get("article"),
            "clauses": chunk.get("clauses", []),
            "page_start": chunk.get("page_start"),
        }
        for chunk in results
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if INDEX_PATH.exists():
        print("Đang tải index đã lưu...")
        chunks, vectorizer, vectors = load_index()
    else:
        print("Đang xây dựng index...")
        chunks, vectorizer, vectors = build_index()
        save_index(chunks, vectorizer, vectors)
        print("Đã lưu index tại:", INDEX_PATH)

    print("Tổng số chunks:", len(chunks))
    print("Tổng số vectors:", len(vectors))
    print("Kích thước vocabulary:", len(vectorizer.vocabulary))

    while True:
        question = input("\nNhập câu hỏi (exit để thoát): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("\nĐang tạo câu trả lời...")
        answer = chat(
            question,
            chunks,
            vectorizer,
            vectors,
        )
        print("\n===== UNIguide AI =====")
        print(answer)
