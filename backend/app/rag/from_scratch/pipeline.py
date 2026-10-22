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
    is_complete_list_question,
    is_tuition_policy_question,
)
from .reranker import rerank_results
from .retriever import reciprocal_rank_fusion, top_k_search


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 15
MAX_CONTEXT_CHARS = 16_000
MAX_CONTEXT_CHUNKS = 16
MAX_GENERATION_ATTEMPTS = 2
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
        results = ask_with_query_processing(
            question,
            chunks,
            vectorizer,
            document_vectors,
        )
    elif retrieval_mode != HYBRID_RETRIEVAL_MODE:
        supported = ", ".join(sorted(SUPPORTED_RETRIEVAL_MODES))
        raise ValueError(
            f"Unsupported retrieval mode: {retrieval_mode}. "
            f"Expected one of: {supported}."
        )
    else:
        active_bm25_index = bm25_index or build_bm25_index(chunks)
        results = ask_hybrid(
            question,
            chunks,
            vectorizer,
            document_vectors,
            active_bm25_index,
        )

    return _expand_same_article_context(question, results, chunks)


def _article_key(chunk: dict) -> tuple[object, object, object] | None:
    """Identify an article without merging equal article numbers across parts."""
    article = chunk.get("article")
    if article is None:
        return None
    return (chunk.get("part"), chunk.get("chapter"), article)


def _chunk_key(chunk: dict) -> tuple[object, ...]:
    """Build a stable identity for in-memory chunks without requiring an id."""
    article_key = _article_key(chunk) or (None, None, None)
    return (
        *article_key,
        tuple(chunk.get("clauses") or []),
        chunk.get("page_start"),
        chunk.get("page_end"),
        chunk.get("text"),
    )


def _tuition_anchor(results: list[dict]) -> dict | None:
    """Prefer the retrieved article that explicitly governs tuition relief."""
    for field in ("article_title", "text"):
        for chunk in results:
            searchable = str(chunk.get(field) or "").casefold()
            if "học phí" in searchable and (
                "miễn" in searchable or "giảm" in searchable
            ):
                return chunk
    return None


def _expand_same_article_context(
    question: str,
    results: list[dict],
    chunks: list[dict],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> list[dict]:
    """Add every chunk of the best-matching article for exhaustive questions."""
    if (
        not results
        or not is_complete_list_question(question)
        or max_chars <= 0
        or max_chunks <= 0
    ):
        return results

    anchor = (
        _tuition_anchor(results)
        if is_tuition_policy_question(question)
        else None
    )
    if anchor is None:
        anchor = next(
            (result for result in results if _article_key(result) is not None),
            None,
        )
    if anchor is None:
        return results

    anchor_key = _article_key(anchor)
    siblings = [chunk for chunk in chunks if _article_key(chunk) == anchor_key]
    candidates = [*siblings, *results]
    expanded: list[dict] = []
    seen: set[tuple[object, ...]] = set()
    used_chars = 0

    for chunk in candidates:
        identity = _chunk_key(chunk)
        if identity in seen:
            continue
        text = str(chunk.get("text") or "")
        if expanded and used_chars + len(text) > max_chars:
            continue
        expanded.append(chunk)
        seen.add(identity)
        used_chars += len(text)
        if len(expanded) >= max_chunks:
            break

    return expanded


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
    answer_requirements = generate_answer_requirements(question)
    validation_error: ValueError | None = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        attempt_requirements = list(answer_requirements)
        if validation_error is not None:
            attempt_requirements.append(
                "Bản trả lời trước không đạt kiểm tra: "
                f"{validation_error}. Hãy sửa đúng lỗi này và tạo lại toàn bộ "
                "câu trả lời từ tài liệu tham khảo."
            )

        grounded = generate_grounded_answer(
            question,
            context,
            history,
            answer_requirements=attempt_requirements,
        )
        try:
            _validate_claim_coverage(
                grounded["answer"], grounded["citations"]
            )
            _validate_required_coverage(question, grounded, results)
            sources = _select_cited_sources(results, grounded["citations"])
        except ValueError as error:
            validation_error = error
            if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                raise
            continue

        return {"answer": grounded["answer"], "sources": sources}

    raise RuntimeError("Không thể tạo câu trả lời hợp lệ")


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


def _claim_level(claim: str) -> str | None:
    """Classify a tuition-policy claim by its stated benefit level."""
    normalized = _normalized_text(claim).replace(" ", "")
    if "70%" in normalized:
        return "70"
    if "50%" in normalized:
        return "50"
    if "100%" in normalized or (
        "miễn" in normalized and "họcphí" in normalized
    ):
        return "100"
    return None


def _validate_required_coverage(
    question: str,
    grounded: dict,
    results: list[dict],
) -> None:
    """Enforce deterministic completeness and scope for known policy lists."""
    if not (
        is_tuition_policy_question(question)
        and is_complete_list_question(question)
    ):
        return

    citations = grounded.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError("Câu trả lời chưa bao phủ các mức miễn, giảm học phí")

    claims: dict[str, str | None] = {}
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("Citation phải là một object")
        claim = citation.get("claim")
        if isinstance(claim, str) and claim.strip():
            claims.setdefault(_normalized_text(claim), _claim_level(claim))

    counts = {
        level: sum(value == level for value in claims.values())
        for level in ("100", "70", "50")
    }
    if counts["100"] < 6 or counts["70"] < 1 or counts["50"] < 1:
        raise ValueError(
            "Câu trả lời phải có đủ 6 nhóm miễn 100%, 1 nhóm giảm 70% "
            "và 1 nhóm giảm 50%"
        )

    anchor = _tuition_anchor(results)
    allowed_key = _article_key(anchor) if anchor is not None else None
    if allowed_key is None:
        return

    for citation in citations:
        document = citation.get("document")
        if (
            not isinstance(document, int)
            or isinstance(document, bool)
            or document < 1
            or document > len(results)
        ):
            raise ValueError("Citation chứa document không hợp lệ")
        if _article_key(results[document - 1]) != allowed_key:
            raise ValueError(
                "Citation về miễn, giảm học phí phải thuộc cùng Điều quy định "
                "miễn, giảm học phí; không được dùng Điều kế bên"
            )


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
                if not evidence_scope and available_clause_set == {clause}:
                    evidence_scope = chunk_text
                elif not evidence_scope:
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
