"""End-to-end indexing pipeline built from the from-scratch components."""

import sys

from app.rag.loader import PDF_PATH, load_pdf

from .bm25 import BM25Index, build_bm25_index
from .chunker import structure_chunk
from .context_builder import build_context
from .embeddings import TfidfVectorizer
from .index_store import INDEX_PATH, load_index, save_index
from .llm import generate_answer, generate_grounded_answer
from .preprocessing import preprocess_pages
from .query_processing import generate_search_queries
from .reranker import rerank_results
from .retriever import reciprocal_rank_fusion, top_k_search


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 15
ORIGINAL_QUERY_WEIGHT = 1.1
HYBRID_RETRIEVAL_MODE = "hybrid"
LEGACY_RETRIEVAL_MODE = "tfidf_qp"
SUPPORTED_RETRIEVAL_MODES = {
    HYBRID_RETRIEVAL_MODE,
    LEGACY_RETRIEVAL_MODE,
}


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
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Run the original single-query TF-IDF retriever."""
    query_vector = vectorizer.transform([question])[0]
    return top_k_search(
        query_vector,
        document_vectors,
        chunks,
        top_k=top_k,
    )


def ask_with_query_processing(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict]:
    """Retrieve with deterministic query expansion and weighted RRF."""
    queries = generate_search_queries(question)
    if len(queries) <= 1:
        return ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
            top_k=top_k,
        )

    ranked_results = []
    for query in queries:
        query_vector = vectorizer.transform([query])[0]
        ranked_results.append(
            top_k_search(
                query_vector,
                document_vectors,
                chunks,
                top_k=candidate_k,
            )
        )

    weights = [ORIGINAL_QUERY_WEIGHT] + [1.0] * (len(queries) - 1)
    return reciprocal_rank_fusion(
        ranked_results,
        weights=weights,
        top_k=top_k,
        source_score_name="tfidf_score",
    )


def ask_bm25(
    question: str,
    chunks: list[dict],
    bm25_index: BM25Index,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Run the original single-query BM25 retriever."""
    return bm25_index.search(question, chunks, top_k=top_k)


def ask_bm25_with_query_processing(
    question: str,
    chunks: list[dict],
    bm25_index: BM25Index,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict]:
    """Retrieve with BM25, deterministic query expansion, and RRF."""
    queries = generate_search_queries(question)
    if len(queries) <= 1:
        return ask_bm25(
            question,
            chunks,
            bm25_index,
            top_k=top_k,
        )

    ranked_results = [
        bm25_index.search(query, chunks, top_k=candidate_k)
        for query in queries
    ]
    weights = [ORIGINAL_QUERY_WEIGHT] + [1.0] * (len(queries) - 1)
    return reciprocal_rank_fusion(
        ranked_results,
        weights=weights,
        top_k=top_k,
        source_score_name="bm25_score",
    )


def ask_hybrid(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    bm25_index: BM25Index,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict]:
    """Fuse query-processed TF-IDF and BM25 rankings."""
    tfidf_results = ask_with_query_processing(
        question,
        chunks,
        vectorizer,
        document_vectors,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )
    bm25_results = ask_bm25_with_query_processing(
        question,
        chunks,
        bm25_index,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )
    fused = reciprocal_rank_fusion(
        [tfidf_results, bm25_results],
        top_k=candidate_k,
        source_score_name="hybrid_source_score",
    )
    search_queries = generate_search_queries(question)
    if len(search_queries) <= 1:
        return fused[:top_k]

    return rerank_results(
        question,
        fused,
        search_queries,
        top_k=top_k,
    )


def chat(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    bm25_index: BM25Index | None = None,
    retrieval_mode: str = HYBRID_RETRIEVAL_MODE,
) -> str:
    """Retrieve relevant chunks and use Ollama to answer the question."""
    results = retrieve_for_chat(
        question,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index=bm25_index,
        retrieval_mode=retrieval_mode,
    )
    if not results:
        return "Không tìm thấy thông tin liên quan trong sổ tay."
    return generate_answer(question, build_context(results))


def retrieve_for_chat(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    bm25_index: BM25Index | None = None,
    retrieval_mode: str = HYBRID_RETRIEVAL_MODE,
) -> list[dict]:
    """Retrieve chat context with Hybrid or the rollback TF-IDF mode."""
    if retrieval_mode == LEGACY_RETRIEVAL_MODE:
        return ask_with_query_processing(
            question,
            chunks,
            vectorizer,
            document_vectors,
        )
    if retrieval_mode != HYBRID_RETRIEVAL_MODE:
        supported = ", ".join(sorted(SUPPORTED_RETRIEVAL_MODES))
        raise ValueError(
            f"Unsupported retrieval mode: {retrieval_mode}. "
            f"Expected one of: {supported}."
        )

    active_bm25_index = bm25_index or build_bm25_index(chunks)
    return ask_hybrid(
        question,
        chunks,
        vectorizer,
        document_vectors,
        active_bm25_index,
    )


def chat_with_sources(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    history: list[dict[str, str]] | None = None,
    bm25_index: BM25Index | None = None,
    retrieval_mode: str = HYBRID_RETRIEVAL_MODE,
) -> dict:
    """Return an answer together with the retrieved source metadata."""
    results = retrieve_for_chat(
        question,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index=bm25_index,
        retrieval_mode=retrieval_mode,
    )
    if not results:
        return {"answer": "Không tìm thấy thông tin liên quan trong sổ tay.", "sources": []}

    context = build_context(results)
    grounded = generate_grounded_answer(question, context, history)
    sources = _select_cited_sources(results, grounded.get("citations", []))
    return {"answer": grounded["answer"], "sources": sources}


def _select_cited_sources(
    results: list[dict],
    citations: list[object],
) -> list[dict]:
    """Map model citations to retrieved chunks without widening their scope."""
    selected: list[dict] = []
    seen: dict[tuple[object, ...], dict] = {}

    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("Citation phải là một object")
        document = citation.get("document")
        clauses = citation.get("clauses")
        if (
            not isinstance(document, int)
            or isinstance(document, bool)
            or not isinstance(clauses, list)
            or not all(isinstance(clause, int) and not isinstance(clause, bool) for clause in clauses)
            or document < 1
            or document > len(results)
        ):
            raise ValueError("Citation chứa document hoặc clauses không hợp lệ")

        chunk = results[document - 1]
        source = {
            "part": chunk.get("part"),
            "chapter": chunk.get("chapter"),
            "article": chunk.get("article"),
            "part_title": chunk.get("part_title"),
            "chapter_title": chunk.get("chapter_title"),
            "article_title": chunk.get("article_title"),
            "clauses": sorted(set(clauses)),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        }
        identity = (
            source["part"],
            source["chapter"],
            source["article"],
            source["page_start"],
            source["page_end"],
        )
        existing = seen.get(identity)
        if existing is not None:
            existing["clauses"] = sorted(
                set(existing["clauses"]) | set(source["clauses"])
            )
            continue
        seen[identity] = source
        selected.append(source)

    return selected


def _select_sources(results: list[dict]) -> list[dict]:
    """Backward-compatible source projection for legacy callers."""
    citations = [
        {"document": index, "clauses": chunk.get("clauses", [])}
        for index, chunk in enumerate(results, start=1)
    ]
    return _select_cited_sources(results, citations)


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

    bm25_index = build_bm25_index(chunks)

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
            bm25_index=bm25_index,
        )
        print("\n===== UNIguide AI =====")
        print(answer)
