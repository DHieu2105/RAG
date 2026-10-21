"""End-to-end indexing pipeline built from the from-scratch components."""

import re
import sys

from app.rag.loader import PDF_PATH, load_pdf

from .bm25 import BM25Index, build_bm25_index
from .chunker import structure_chunk
from .context_builder import build_context
from .embeddings import TfidfVectorizer
from .index_store import INDEX_PATH, load_index, save_index
from .llm import generate_answer, generate_grounded_answer
from .preprocessing import preprocess_pages
from .query_processing import (
    generate_answer_requirements,
    generate_search_queries,
)
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
    grounded = generate_grounded_answer(
        question,
        context,
        history,
        answer_requirements=generate_answer_requirements(question),
    )
    _validate_claim_coverage(grounded["answer"], grounded["citations"])
    sources = _select_cited_sources(results, grounded["citations"])
    return {"answer": grounded["answer"], "sources": sources}


def _normalized_text(value: str) -> str:
    """Normalize whitespace for exact claim-to-answer coverage checks."""
    return " ".join(value.casefold().split())


def _normalized_evidence(value: str) -> str:
    """Normalize typographic punctuation without changing evidence words."""
    punctuation = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "„": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
        }
    )
    return _normalized_text(value.translate(punctuation))


def _clause_sections(text: str) -> dict[int, str]:
    """Split a structured chunk into its top-level numbered clauses."""
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", text))
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        sections[int(match.group(1))] = text[match.start():end].strip()
    return sections


def _answer_sentences(answer: str) -> list[str]:
    """Return non-empty answer sentences used by citation coverage checks."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if sentence.strip()
    ]


def _validate_claim_coverage(answer: str, citations: list[object]) -> None:
    """Require every answer sentence to be represented by a cited claim."""
    if not citations:
        return

    normalized_answer = _normalized_text(answer)
    normalized_claims: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("Citation phải là một object")
        claim = citation.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Citation thiếu claim hợp lệ")
        normalized_claim = _normalized_text(claim)
        if normalized_claim not in normalized_answer:
            raise ValueError("Claim được trích dẫn không xuất hiện trong answer")
        normalized_claims.append(normalized_claim)

    for sentence in _answer_sentences(answer):
        normalized_sentence = _normalized_text(sentence)
        if not any(
            normalized_sentence in claim or claim in normalized_sentence
            for claim in normalized_claims
        ):
            raise ValueError("Answer chứa một ý chưa được gắn citation")


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
        claim = citation.get("claim")
        evidence = citation.get("evidence")
        legacy_projection = claim is None and evidence is None
        if (
            not isinstance(document, int)
            or isinstance(document, bool)
            or not isinstance(clauses, list)
            or not all(
                isinstance(clause, int) and not isinstance(clause, bool)
                for clause in clauses
            )
            or document < 1
            or document > len(results)
            or (
                not legacy_projection
                and (
                    not isinstance(evidence, str)
                    or not evidence.strip()
                )
            )
        ):
            raise ValueError("Citation chứa document hoặc clauses không hợp lệ")

        chunk = results[document - 1]
        available_clauses = chunk.get("clauses")
        if available_clauses is None:
            available_clause = chunk.get("clause")
            available_clauses = (
                [available_clause] if available_clause is not None else []
            )
        available_clause_set = set(available_clauses)
        cited_clause_set = set(clauses)
        if (
            (available_clause_set and not cited_clause_set)
            or not cited_clause_set.issubset(available_clause_set)
            or (not legacy_projection and len(cited_clause_set) > 1)
        ):
            raise ValueError(
                "Citation chứa khoản không thuộc tài liệu được viện dẫn"
            )

        if not legacy_projection:
            chunk_text = chunk.get("text")
            if not isinstance(chunk_text, str):
                raise ValueError("Tài liệu được viện dẫn không có nội dung")
            evidence_scope = chunk_text
            if cited_clause_set:
                clause = next(iter(cited_clause_set))
                evidence_scope = _clause_sections(chunk_text).get(clause, "")
                if not evidence_scope:
                    raise ValueError(
                        "Không xác định được nội dung của khoản viện dẫn"
                    )
            if _normalized_evidence(evidence) not in _normalized_evidence(
                evidence_scope
            ):
                raise ValueError(
                    "Evidence không xuất hiện trong đúng tài liệu và khoản viện dẫn"
                )

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
            "claims": [claim] if isinstance(claim, str) and claim.strip() else [],
            "evidence": (
                [evidence.strip()] if isinstance(evidence, str) else []
            ),
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
            for source_claim in source["claims"]:
                if source_claim not in existing["claims"]:
                    existing["claims"].append(source_claim)
            for source_evidence in source["evidence"]:
                if source_evidence not in existing["evidence"]:
                    existing["evidence"].append(source_evidence)
            continue
        seen[identity] = source
        selected.append(source)

    return selected


def _select_sources(results: list[dict]) -> list[dict]:
    """Backward-compatible source projection for legacy callers."""
    citations = [
        {
            "document": index,
            "clauses": chunk.get("clauses", []),
        }
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
