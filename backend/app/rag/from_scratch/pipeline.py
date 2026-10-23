"""End-to-end indexing pipeline built from the from-scratch components."""

import re
import sys
from enum import Enum

from app.rag.loader import PDF_PATH, load_pdf

from .bm25 import BM25Index, build_bm25_index
from .chunker import structure_chunk
from .context_builder import build_context
from .embeddings import TfidfVectorizer
from .grounding import (
    AnswerCoverageError,
    CitationValidationError,
    context_is_relevant,
    coverage_requirements,
    extract_coverage_units,
    prepare_structured_context,
    validate_answer_coverage,
    validate_claim_faithfulness,
)
from .index_store import INDEX_PATH, load_index, save_index
from .llm import generate_answer, generate_grounded_answer
from .preprocessing import preprocess_pages
from .query_processing import (
    generate_answer_requirements,
    generate_search_queries,
)
from .query_understanding import analyze_query
from .reranker import rerank_results
from .retriever import reciprocal_rank_fusion, top_k_search


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 15
CHAT_TOP_K = 8
MAX_CONTEXT_CHARS = 16_000
MAX_CONTEXT_CHUNKS = 16
MAX_GENERATION_ATTEMPTS = 2
ORIGINAL_QUERY_WEIGHT = 1.1
NO_RELEVANT_DOCUMENT_ANSWER = "Không tìm thấy tài liệu liên quan trong sổ tay."
INCOMPLETE_CONTEXT_ANSWER = (
    "Đã tìm thấy tài liệu liên quan nhưng ngữ cảnh truy xuất chưa đủ để trả lời chắc chắn."
)
INCOMPLETE_GENERATION_ANSWER = (
    "Đã tìm thấy căn cứ liên quan nhưng câu trả lời sinh ra chưa bao phủ đầy đủ thông tin."
)
INVALID_CITATION_ANSWER = (
    "Đã tạo câu trả lời nhưng trích dẫn chưa hợp lệ nên chưa thể xác minh."
)


class AnswerState(str, Enum):
    NO_RELEVANT_DOCUMENT = "no_relevant_document"
    INCOMPLETE_CONTEXT = "incomplete_context"
    INCOMPLETE_GENERATION = "incomplete_generation"
    INVALID_CITATION = "invalid_citation"
    VERIFIED = "verified"


def classify_answer_state(answer: str) -> AnswerState:
    """Map the unchanged API answer field to an explicit internal state."""
    states = {
        NO_RELEVANT_DOCUMENT_ANSWER: AnswerState.NO_RELEVANT_DOCUMENT,
        INCOMPLETE_CONTEXT_ANSWER: AnswerState.INCOMPLETE_CONTEXT,
        INCOMPLETE_GENERATION_ANSWER: AnswerState.INCOMPLETE_GENERATION,
        INVALID_CITATION_ANSWER: AnswerState.INVALID_CITATION,
    }
    return states.get(answer, AnswerState.VERIFIED)


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
        return NO_RELEVANT_DOCUMENT_ANSWER
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
    analysis = analyze_query(question)
    chat_top_k = CHAT_TOP_K if analysis.requires_complete_coverage else DEFAULT_TOP_K
    if retrieval_mode == LEGACY_RETRIEVAL_MODE:
        results = ask_with_query_processing(
            question,
            chunks,
            vectorizer,
            document_vectors,
            top_k=chat_top_k,
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
            top_k=chat_top_k,
        )

    return _expand_same_article_context(question, results, chunks)


def _expand_same_article_context(
    question: str,
    results: list[dict],
    chunks: list[dict],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> list[dict]:
    """Expand and filter context by generic intent and document structure."""
    if max_chars <= 0 or max_chunks <= 0:
        return []
    return prepare_structured_context(
        analyze_query(question),
        results,
        chunks,
        max_chars=max_chars,
        max_chunks=max_chunks,
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
    analysis = analyze_query(question)
    results = retrieve_for_chat(
        question,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index=bm25_index,
        retrieval_mode=retrieval_mode,
    )
    known_terms = getattr(vectorizer, "vocabulary", None)
    if isinstance(known_terms, dict):
        known_terms = set(known_terms)
    else:
        known_terms = None
    if not context_is_relevant(analysis, results, known_terms=known_terms):
        return {"answer": NO_RELEVANT_DOCUMENT_ANSWER, "sources": []}
    if any(not result.get("_context_complete", True) for result in results):
        return {"answer": INCOMPLETE_CONTEXT_ANSWER, "sources": []}

    context = build_context(results)
    answer_requirements = generate_answer_requirements(question)
    coverage_units = extract_coverage_units(analysis, results)
    answer_requirements.extend(coverage_requirements(coverage_units))
    validation_error: ValueError | None = None
    final_failure = "generation"

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        attempt_requirements = list(answer_requirements)
        if validation_error is not None:
            attempt_requirements.append(
                "Bản trả lời trước không đạt kiểm tra: "
                f"{validation_error}. Hãy sửa đúng lỗi này và tạo lại toàn bộ "
                "câu trả lời từ tài liệu tham khảo."
            )

        try:
            grounded = generate_grounded_answer(
                question,
                context,
                history,
                answer_requirements=attempt_requirements,
            )
            _normalize_citation_metadata(results, grounded.get("citations", []))
            if not grounded.get("citations"):
                raise AnswerCoverageError(
                    "Model không tạo được claim có citation từ context"
                )
            _validate_claim_coverage(
                grounded["answer"], grounded["citations"]
            )
            sources = _select_cited_sources(results, grounded["citations"])
            validate_claim_faithfulness(
                analysis,
                results,
                grounded["citations"],
            )
            validate_answer_coverage(coverage_units, grounded["citations"])
        except AnswerCoverageError as error:
            validation_error = error
            final_failure = "generation"
            if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                return {"answer": INCOMPLETE_GENERATION_ANSWER, "sources": []}
            continue
        except (CitationValidationError, ValueError) as error:
            validation_error = error
            final_failure = "citation"
            if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                return {"answer": INVALID_CITATION_ANSWER, "sources": []}
            continue

        return {"answer": grounded["answer"], "sources": sources}

    if final_failure == "citation":
        return {"answer": INVALID_CITATION_ANSWER, "sources": []}
    return {"answer": INCOMPLETE_GENERATION_ANSWER, "sources": []}


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


def _normalize_citation_metadata(
    results: list[dict],
    citations: list[object],
) -> None:
    """Repair only clause labels disproved by exact evidence metadata.

    Qwen occasionally invents a clause number for an unnumbered paragraph.
    When the quoted evidence is an exact span of that paragraph, clearing the
    clause is deterministic and more accurate than rejecting the citation.
    """
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        document = citation.get("document")
        clauses = citation.get("clauses")
        evidence = citation.get("evidence")
        if (
            not isinstance(document, int)
            or isinstance(document, bool)
            or not 1 <= document <= len(results)
            or not isinstance(clauses, list)
            or not clauses
            or not isinstance(evidence, str)
        ):
            continue
        chunk = results[document - 1]
        available_clauses = chunk.get("clauses") or []
        text = str(chunk.get("text") or "")
        normalized_text = _normalized_evidence(text)
        if _normalized_evidence(evidence) not in normalized_text:
            # Small models sometimes prepend the shared list introduction to
            # item b/c even though item a lies between them in the source.
            # Keep only the longest exact item suffix; never invent wording.
            item_starts = [
                match.start()
                for match in re.finditer(
                    r"(?i)(?<!\w)(?:[a-zđ]\)|[a-zđ]\.\s|bước\s+\d+\s*:)",
                    evidence,
                )
            ]
            exact_suffixes = [
                evidence[start:].strip()
                for start in item_starts
                if _normalized_evidence(evidence[start:].strip()) in normalized_text
            ]
            if exact_suffixes:
                citation["evidence"] = max(exact_suffixes, key=len)
                evidence = citation["evidence"]
        if not available_clauses and _normalized_evidence(
            evidence
        ) in _normalized_evidence(text):
            citation["clauses"] = []


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


def _validate_required_coverage(
    question: str,
    grounded: dict,
    results: list[dict],
) -> None:
    """Validate completeness from information units extracted from context."""
    analysis = analyze_query(question)
    units = extract_coverage_units(analysis, results)
    validate_answer_coverage(units, grounded.get("citations", []))


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
            normalized_evidence = _normalized_evidence(evidence)
            evidence_matches_scope = normalized_evidence in _normalized_evidence(
                evidence_scope
            )
            if (
                not evidence_matches_scope
                and cited_clause_set
                and available_clause_set == cited_clause_set
            ):
                evidence_matches_scope = (
                    normalized_evidence in _normalized_evidence(chunk_text)
                )
            if not evidence_matches_scope:
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
