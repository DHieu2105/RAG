"""End-to-end indexing pipeline built from the from-scratch components."""

import os
import re
import sys
from typing import Any
import unicodedata
from enum import Enum

from app.rag.loader import PDF_PATH, load_pdf

from .bm25 import BM25Index, build_bm25_index
from .chunker import structure_chunk
from .context_builder import build_context
from .embeddings import TfidfVectorizer
from .grounding import (
    AnswerCoverageError,
    CitationValidationError,
    _merge_same_scope_fragments,
    context_is_relevant,
    coverage_requirements,
    extract_coverage_units,
    prepare_structured_context,
    validate_answer_coverage,
    validate_answer_style,
    validate_claim_faithfulness,
)
from .index_store import INDEX_PATH, load_index, save_index
from .llm import generate_answer
from .preprocessing import preprocess_pages
from .query_processing import (
    generate_answer_requirements,
    generate_search_queries,
)
from .query_understanding import QueryAnalysis, analyze_query, lexical_terms
from .reranker import rerank_results
from .retriever import reciprocal_rank_fusion, top_k_search
from .tokenizer import tokenize


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 15
CHAT_TOP_K = 8
MAX_CONTEXT_CHARS = 16_000
MAX_CONTEXT_CHUNKS = 16
MAX_GENERATION_ATTEMPTS = 2
ORIGINAL_QUERY_WEIGHT = 1.1
DEBUG_RAG = os.getenv("RAG_DEBUG", "0").lower() in ("1", "true", "yes")
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
    refusal_phrases = (
        "không tìm thấy thông tin liên quan",
        "không có thông tin liên quan",
        "không tìm thấy tài liệu liên quan",
    )
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return AnswerState.NO_RELEVANT_DOCUMENT
    if "chưa đủ thông tin để trả lời chắc chắn" in answer_lower:
        return AnswerState.INCOMPLETE_CONTEXT
    if INCOMPLETE_GENERATION_ANSWER.lower() in answer_lower:
        return AnswerState.INCOMPLETE_GENERATION
    if INVALID_CITATION_ANSWER.lower() in answer_lower:
        return AnswerState.INVALID_CITATION

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
    *,
    use_query_processing: bool = False,
    use_reranking: bool = False,
) -> list[dict]:
    """Fuse TF-IDF and BM25 rankings.

    By default in the simplified production path, query expansion and reranking
    are removed to keep the user query unchanged and avoid noisy alias drifts.
    """
    if use_query_processing:
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
    else:
        tfidf_results = ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
            top_k=candidate_k,
        )
        bm25_results = ask_bm25(
            question,
            chunks,
            bm25_index,
            top_k=candidate_k,
        )

    fused = reciprocal_rank_fusion(
        [tfidf_results, bm25_results],
        top_k=candidate_k if use_reranking else top_k,
        source_score_name="hybrid_source_score",
    )
    if not use_reranking:
        return fused[:top_k]

    search_queries = generate_search_queries(question) if use_query_processing else [question]
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
    """Retrieve relevant chunks and use Gemini to answer the question."""
    results = retrieve_for_chat(
        question,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index=bm25_index,
        retrieval_mode=retrieval_mode,
    )
    if DEBUG_RAG:
        print("\n========== DEBUG RETRIEVAL ==========")
        for index, chunk in enumerate(results, start=1):
            print(
                f"\n[Tài liệu {index}] "
                f"Phần {chunk.get('part')} | "
                f"Điều {chunk.get('article')} | "
                f"Khoản {chunk.get('clauses')} | "
                f"Trang {chunk.get('page_start')}-{chunk.get('page_end')}"
            )
            print(chunk.get("text", ""))
        print("\n========== END DEBUG RETRIEVAL ==========\n")
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
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Retrieve chat context using the chosen retrieval mode."""
    if retrieval_mode == LEGACY_RETRIEVAL_MODE:
        results = ask_with_query_processing(
            question,
            chunks,
            vectorizer,
            document_vectors,
            top_k=top_k,
        )
    elif retrieval_mode == "tfidf":
        results = ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
            top_k=top_k,
        )
    elif retrieval_mode == "bm25":
        active_bm25_index = bm25_index or build_bm25_index(chunks)
        results = ask_bm25(
            question,
            chunks,
            active_bm25_index,
            top_k=top_k,
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
            top_k=top_k,
        )

    for result in results:
        if "_context_complete" not in result:
            result["_context_complete"] = True

    if DEBUG_RAG:
        print("\n========== RAW RETRIEVAL ==========")
        for index, chunk in enumerate(results, start=1):
            print(
                f"[RAW {index}] "
                f"Score: {chunk.get('score')} | "
                f"Điều {chunk.get('article')} | "
                f"Khoản {chunk.get('clauses')} | "
                f"{chunk.get('article_title')}"
            )
        print("========== END RAW RETRIEVAL ==========\n")

    return results


def _retrieval_identity(chunk: dict) -> tuple[object, ...]:
    return (
        chunk.get("part"),
        chunk.get("chapter"),
        chunk.get("article"),
        tuple(chunk.get("clauses") or []),
        chunk.get("page_start"),
        chunk.get("page_end"),
        chunk.get("text"),
    )


def _add_title_matches(
    analysis: QueryAnalysis,
    results: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """Seed strong article-title matches missed by content-only indexes."""
    focus = set(analysis.focus_terms)
    if not focus:
        return results

    scored: list[tuple[dict, float]] = []
    for chunk in chunks:
        title = str(chunk.get("article_title") or "")
        if not title:
            continue
        title_terms = set(lexical_terms(title))
        coverage = len(title_terms & focus) / len(focus)
        scored.append((chunk, coverage))

    best_coverage = max((coverage for _, coverage in scored), default=0.0)
    if best_coverage < 0.75:
        return results

    threshold = max(0.75, best_coverage * 0.9)
    augmented = list(results)
    seen = {_retrieval_identity(result) for result in results}
    for chunk, coverage in scored:
        identity = _retrieval_identity(chunk)
        if coverage >= threshold and identity not in seen:
            augmented.append(chunk)
            seen.add(identity)
    return augmented


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


def _prepare_chat_context(
    results: list[dict],
    all_chunks: list[dict],
) -> list[dict]:
    """Reassemble split consecutive fragments of the same scope into continuous text."""
    if not results or not all_chunks:
        return results

    def _scope(c: dict) -> tuple[object, ...]:
        return (
            c.get("part"),
            c.get("chapter"),
            c.get("article"),
            tuple(c.get("clauses") or []),
        )

    def _chunk_id(c: dict) -> tuple[object, ...]:
        return (
            c.get("part"),
            c.get("chapter"),
            c.get("article"),
            tuple(c.get("clauses") or []),
            c.get("page_start"),
            c.get("page_end"),
            c.get("text"),
        )

    chunk_to_index = {_chunk_id(c): idx for idx, c in enumerate(all_chunks)}
    ordered_chunks: list[dict] = []
    seen_indices: set[int] = set()

    for r in results:
        idx = chunk_to_index.get(_chunk_id(r))
        if idx is None:
            ordered_chunks.append(r)
            continue

        r_scope = _scope(r)
        start = idx
        while start > 0 and _scope(all_chunks[start - 1]) == r_scope:
            start -= 1
        end = idx
        while end < len(all_chunks) - 1 and _scope(all_chunks[end + 1]) == r_scope:
            end += 1

        for i in range(start, end + 1):
            if i not in seen_indices:
                seen_indices.add(i)
                ordered_chunks.append(all_chunks[i])

    return _merge_same_scope_fragments(ordered_chunks)


def _parse_claim_citations(raw_line: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract metadata citations from line and return clean claim text."""
    cite_matches = list(
        re.finditer(
            r"[\*\s]*\(([^)]*(?:Điều|Khoản|Phần|Trang|Tài liệu)[^)]*)\)[\*\s]*",
            raw_line,
            re.IGNORECASE,
        )
    )
    citations: list[dict[str, Any]] = []
    for m in cite_matches:
        s = m.group(1)
        info: dict[str, Any] = {}
        art_m = re.search(r"Điều\s+(\d+)", s, re.IGNORECASE)
        if art_m:
            info["article"] = int(art_m.group(1))
        cl_m = re.search(r"Khoản\s+(\d+)", s, re.IGNORECASE)
        if cl_m:
            info["clause"] = int(cl_m.group(1))
        p_m = re.search(r"Phần\s+([0-9.]+)", s, re.IGNORECASE)
        if p_m:
            info["part"] = p_m.group(1)
        pg_m = re.search(r"Trang\s+(?:PDF\s+)?(\d+)", s, re.IGNORECASE)
        if pg_m:
            info["page"] = int(pg_m.group(1))
        doc_m = re.search(r"Tài liệu\s+(\d+)", s, re.IGNORECASE)
        if doc_m:
            info["document"] = int(doc_m.group(1))
        citations.append(info)

    clean = re.sub(
        r"[\*\s]*\(([^)]*(?:Điều|Khoản|Phần|Trang|Tài liệu)[^)]*)\)[\*\s]*",
        " ",
        raw_line,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" .;,*-")
    return clean, citations


def _clean_tokens(text: str, stopwords: set[str]) -> set[str]:
    """Extract lowercase tokens stripped of punctuation and stopwords."""
    tokens = set()
    for w in tokenize(text.lower()):
        cleaned = w.strip(".,;:?!\"'()[]{}-*")
        if cleaned and cleaned not in stopwords and len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def _extract_best_evidence_sentence(
    chunk_text: str,
    claim_tokens: set[str],
    stopwords: set[str],
    clean_claim: str = "",
) -> str:
    """Find the sentence in chunk_text with the strongest support for the claim."""
    pattern = r"[\n\r]+|(?<!\b\d)(?<!\b[a-zA-Z])[.!?]\s+"
    sentences = [
        s.strip()
        for s in re.split(pattern, chunk_text)
        if s.strip()
    ]
    if not sentences:
        return chunk_text

    # Extract 2-word shingles from clean_claim for phrase bonus
    claim_words = (
        [
            w.strip(".,;:?!\"'()[]{}-*")
            for w in tokenize(clean_claim.lower())
            if w.strip(".,;:?!\"'()[]{}-*") not in stopwords
            and len(w.strip(".,;:?!\"'()[]{}-*")) > 1
        ]
        if clean_claim
        else []
    )
    shingles = [" ".join(claim_words[i:i + 2]) for i in range(len(claim_words) - 1)]

    best_sent = sentences[0]
    best_score = -1.0
    for s in sentences:
        s_tokens = _clean_tokens(s, stopwords)
        if not s_tokens:
            continue
        overlap = len(claim_tokens & s_tokens)
        s_lower = s.lower()
        shingle_score = sum(3.0 for sh in shingles if sh in s_lower)
        raw_score = float(overlap) + shingle_score
        # Length normalization to prevent verbose boilerplate sentences from beating short focused evidence
        norm_score = raw_score / (len(s_tokens) ** 0.4)
        if norm_score > best_score:
            best_score = norm_score
            best_sent = s
    return best_sent


def _filter_supporting_sources(results: list[dict], answer: str) -> list[dict]:
    """Verify evidence against candidate chunks and return verified source citations.

    Never falls back to all results when unverified. Sources must have verified
    evidence supporting an actual claim from the answer.
    """
    if not results or not answer:
        return []

    refusal_phrases = (
        "không tìm thấy thông tin liên quan",
        "không có thông tin liên quan",
        "không tìm thấy tài liệu liên quan",
        "tài liệu được cung cấp chưa đủ thông tin",
    )
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return []

    stopwords = {
        "sinh", "viên", "học", "nhà", "trường", "được", "các", "cho", "của", "trong",
        "theo", "và", "hoặc", "có", "phải", "tại", "với", "về", "sau", "đây", "để",
        "trên", "khi", "nếu", "thì", "một", "những", "người", "quy", "định", "phần",
        "điều", "khoản", "trang", "sổ", "tay", "utc", "k67", "vay", "vốn",
    }

    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    claims_to_check: list[str] = []
    for l in lines:
        if l.startswith(("- ", "* ")):
            claims_to_check.append(l[2:].strip())
        elif re.match(r"^\d+\.\s+", l):
            claims_to_check.append(re.sub(r"^\d+\.\s+", "", l).strip())
    if not claims_to_check:
        claims_to_check = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]

    verified_sources: list[dict] = []
    seen: dict[tuple[object, ...], dict] = {}

    for raw_claim in claims_to_check:
        clean_claim, citations = _parse_claim_citations(raw_claim)
        if not clean_claim:
            continue

        claim_tokens = _clean_tokens(clean_claim, stopwords)
        claim_numbers = set(
            re.findall(r"(?<![A-Za-z0-9_/-])\d+(?:[.,]\d+)?%?(?![A-Za-z0-9_/-])", clean_claim)
        )

        has_explicit_article = any("article" in cite for cite in citations)
        has_explicit_doc = any("document" in cite for cite in citations)

        matching_chunks: list[tuple[float, dict, str]] = []
        for idx, chunk in enumerate(results, start=1):
            c_text = str(chunk.get("text") or "")
            c_tokens = _clean_tokens(c_text, stopwords)
            overlap = len(claim_tokens & c_tokens)

            matched_cite = False
            for cite in citations:
                if cite.get("document") == idx:
                    matched_cite = True
                    break
                if "article" in cite and chunk.get("article") == cite["article"]:
                    if "clause" in cite and chunk.get("clauses"):
                        if cite["clause"] in chunk["clauses"]:
                            matched_cite = True
                            break
                    else:
                        matched_cite = True
                        break
                elif "clause" in cite and chunk.get("clauses") and cite["clause"] in chunk["clauses"]:
                    if "page" in cite:
                        page_start = chunk.get("page_start")
                        page_end = chunk.get("page_end") or page_start
                        if page_start is not None and page_start <= cite["page"] <= page_end:
                            matched_cite = True
                            break
                    else:
                        matched_cite = True
                        break

            # If the claim explicitly cited a specific article or document,
            # only consider chunks matching that citation.
            if (has_explicit_article or has_explicit_doc) and not matched_cite:
                continue

            num_ok = True
            if claim_numbers:
                c_num_str = c_text.replace(",", ".")
                has_any_num = any(
                    num in c_text or num.replace(",", ".") in c_num_str
                    for num in claim_numbers
                )
                if not has_any_num:
                    num_ok = False

            evidence = _extract_best_evidence_sentence(
                c_text, claim_tokens, stopwords, clean_claim=clean_claim
            )
            s_tokens = _clean_tokens(evidence, stopwords)
            sent_overlap = len(claim_tokens & s_tokens)

            # Shingle check for 2 or 3 word sequences
            shingle_match = False
            claim_words = [
                w.strip(".,;:?!\"'()[]{}-*")
                for w in tokenize(clean_claim.lower())
                if w.strip(".,;:?!\"'()[]{}-*") not in stopwords
                and len(w.strip(".,;:?!\"'()[]{}-*")) > 1
            ]
            c_lower = c_text.lower()
            for i in range(len(claim_words) - 1):
                shingle = " ".join(claim_words[i:i + 2])
                if shingle in c_lower:
                    shingle_match = True
                    break

            is_supported = False
            if matched_cite:
                # Even with explicit citation, chunk must contain semantic evidence and consistent numbers
                if num_ok and overlap >= 2 and sent_overlap >= 1:
                    is_supported = True
            else:
                # Without explicit citation, require stronger lexical/shingle support
                if num_ok and sent_overlap >= 1 and ((overlap >= 3 and shingle_match) or overlap >= 4):
                    is_supported = True

            if is_supported:
                score = (100.0 if matched_cite else 0.0) + float(overlap) + float(sent_overlap)
                matching_chunks.append((score, chunk, evidence))

        if matching_chunks:
            matching_chunks.sort(key=lambda x: x[0], reverse=True)
            _, best_chunk, best_evidence = matching_chunks[0]

            article_title = best_chunk.get("article_title")
            if not article_title:
                if best_chunk.get("article") is not None:
                    article_title = f"Điều {best_chunk.get('article')}"
                elif best_chunk.get("chapter_title"):
                    article_title = best_chunk.get("chapter_title")
                elif best_chunk.get("part_title"):
                    article_title = best_chunk.get("part_title")
                else:
                    article_title = "Quy định liên quan"

            key = (
                best_chunk.get("part"),
                best_chunk.get("chapter"),
                best_chunk.get("article"),
                best_chunk.get("page_start"),
                best_chunk.get("page_end"),
            )
            if key in seen:
                src = seen[key]
                if clean_claim not in src["claims"]:
                    src["claims"].append(clean_claim)
                if best_evidence not in src["evidence"]:
                    src["evidence"].append(best_evidence)
            else:
                src = {
                    "part": best_chunk.get("part"),
                    "chapter": best_chunk.get("chapter"),
                    "article": best_chunk.get("article"),
                    "part_title": best_chunk.get("part_title"),
                    "chapter_title": best_chunk.get("chapter_title"),
                    "article_title": article_title,
                    "clauses": sorted(set(best_chunk.get("clauses") or [])),
                    "page_start": best_chunk.get("page_start"),
                    "page_end": best_chunk.get("page_end"),
                    "claims": [clean_claim],
                    "evidence": [best_evidence],
                }
                seen[key] = src
                verified_sources.append(src)

    return verified_sources


def chat_with_sources(
    question: str,
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    history: list[dict[str, str]] | None = None,
    bm25_index: BM25Index | None = None,
    retrieval_mode: str = HYBRID_RETRIEVAL_MODE,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Simplified 5-step RAG workflow:
    1. User enters query.
    2. Retriever finds relevant chunks from index.
    3. Context Builder augments chunks into prompt.
    4. LLM receives prompt and generates grounded answer.
    5. Return answer together with source citations.
    """
    results = retrieve_for_chat(
        question,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index=bm25_index,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
    )

    if not results:
        return {"answer": NO_RELEVANT_DOCUMENT_ANSWER, "sources": []}

    if any(not result.get("_context_complete", True) for result in results):
        return {"answer": INCOMPLETE_CONTEXT_ANSWER, "sources": []}

    context_chunks = _prepare_chat_context(results, chunks)

    if DEBUG_RAG:
        print("\n========== DEBUG CONTEXT CHUNKS ==========")
        for index, chunk in enumerate(context_chunks, start=1):
            print(
                f"\n[Tài liệu {index}] "
                f"Phần {chunk.get('part')} | "
                f"Điều {chunk.get('article')} | "
                f"Khoản {chunk.get('clauses')} | "
                f"Trang {chunk.get('page_start')}-{chunk.get('page_end')}"
            )
            print(chunk.get("text", ""))
        print("\n========== END DEBUG CONTEXT CHUNKS ==========\n")

    context = build_context(context_chunks)
    if DEBUG_RAG:
        print("\n========== DEBUG CONTEXT SENT TO LLM ==========")
        print(context)
        print("========== END DEBUG CONTEXT SENT TO LLM ==========\n")
    answer = generate_answer(question, context, history=history)

    refusal_phrases = (
        "không tìm thấy thông tin liên quan",
        "không có thông tin liên quan",
        "không tìm thấy tài liệu liên quan",
        "tài liệu được cung cấp chưa đủ thông tin để trả lời chắc chắn",
    )
    is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)

    if is_refusal:
        sources = []
    else:
        sources = _filter_supporting_sources(context_chunks, answer)

    return {"answer": answer, "sources": sources}


def _normalized_text(value: str) -> str:
    """Normalize whitespace for exact claim-to-answer coverage checks."""
    return " ".join(value.casefold().split())


def _normalized_evidence(value: str) -> str:
    """Normalize Unicode and typographic punctuation for evidence checks."""
    value = unicodedata.normalize("NFC", value)
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

        evidence_list = []
        if isinstance(evidence, str) and evidence.strip():
            evidence_list = [evidence.strip()]
        elif chunk.get("text") and str(chunk["text"]).strip():
            evidence_list = [str(chunk["text"]).strip()]

        article_title = chunk.get("article_title")
        if not article_title:
            if chunk.get("article") is not None:
                article_title = f"Điều {chunk.get('article')}"
            elif chunk.get("chapter_title"):
                article_title = chunk.get("chapter_title")
            elif chunk.get("part_title"):
                part_num = chunk.get("part")
                article_title = f"Phần {part_num}: {chunk.get('part_title')}" if part_num else chunk.get("part_title")
            elif chunk.get("part"):
                article_title = f"Phần {chunk.get('part')}"
            else:
                article_title = "Quy định liên quan"

        source = {
            "part": chunk.get("part"),
            "chapter": chunk.get("chapter"),
            "article": chunk.get("article"),
            "part_title": chunk.get("part_title"),
            "chapter_title": chunk.get("chapter_title"),
            "article_title": article_title,
            "clauses": sorted(set(clauses)),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "claims": [claim] if isinstance(claim, str) and claim.strip() else [],
            "evidence": evidence_list,
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
