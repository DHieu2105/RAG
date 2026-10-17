"""End-to-end indexing pipeline built from the from-scratch components."""

from app.rag.loader import PDF_PATH, load_pdf

from .chunker import structure_chunk
from .embeddings import TfidfVectorizer
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


if __name__ == "__main__":
    print("Đang xây dựng index...")

    chunks, vectorizer, vectors = build_index()

    print("Tổng số chunks:", len(chunks))
    print("Tổng số vectors:", len(vectors))
    print("Kích thước vocabulary:", len(vectorizer.vocabulary))

    question = "Sinh viên bị cảnh báo học tập khi nào?"

    print("\nCâu hỏi:", question)

    results = ask(
        question,
        chunks,
        vectorizer,
        vectors,
    )

    for index, result in enumerate(results, start=1):
        print(f"\n===== KẾT QUẢ {index} =====")
        print("Điều:", result.get("article"))
        print("Khoản:", result.get("clauses", result.get("clause")))
        print("Trang:", result.get("page_start"))
        print("Score:", round(result["score"], 4))
        print("Nội dung:", result["text"])
