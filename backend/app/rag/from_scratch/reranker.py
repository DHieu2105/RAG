"""Deterministic lexical reranking for fused retrieval candidates."""

import re


RANK_WEIGHT = 3.0
COVERAGE_WEIGHT = 0.25
ANSWER_CUE_WEIGHT = 0.10
RANK_OFFSET = 10

STOP_WORDS = {
    "ai",
    "bao",
    "bị",
    "có",
    "của",
    "gì",
    "khi",
    "không",
    "là",
    "nào",
    "những",
    "quá",
    "sinh",
    "thì",
    "viên",
    "và",
    "được",
}

CONDITION_QUESTION_CUES = (
    "khi nào",
    "điều kiện",
    "trường hợp",
    "có bị",
)
CONDITION_ANSWER_CUES = (
    "các trường hợp sau",
    "trong trường hợp",
    "đủ các điều kiện",
    "điều kiện sau",
    "thuộc diện",
    "nếu ",
)


def lexical_terms(text: str) -> set[str]:
    """Extract comparable Unicode word terms for reranking only."""
    terms = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    return {term for term in terms if term not in STOP_WORDS}


def _coverage(text_terms: set[str], queries: list[str]) -> float:
    query_coverages: list[float] = []
    for query in queries:
        query_terms = lexical_terms(query)
        if query_terms:
            query_coverages.append(
                len(text_terms & query_terms) / len(query_terms)
            )
    if not query_coverages:
        return 0.0
    return sum(query_coverages) / len(query_coverages)


def _answer_cue_score(question: str, text: str) -> float:
    folded_question = question.casefold()
    folded_text = text.casefold()
    asks_for_condition = any(
        cue in folded_question for cue in CONDITION_QUESTION_CUES
    )
    if not asks_for_condition:
        return 0.0

    matches = sum(
        cue in folded_text for cue in CONDITION_ANSWER_CUES
    )
    return min(matches / 2.0, 1.0)


def rerank_results(
    question: str,
    candidates: list[dict],
    search_queries: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Rerank candidates by rank prior, query coverage, and answer cues."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not candidates:
        return []

    reranked: list[dict] = []

    for index, candidate in enumerate(candidates):
        rank_prior = 1.0 / (RANK_OFFSET + index + 1)
        text_terms = lexical_terms(candidate.get("text", ""))
        coverage = _coverage(text_terms, search_queries)
        answer_cue = _answer_cue_score(
            question,
            candidate.get("text", ""),
        )
        rerank_score = (
            RANK_WEIGHT * rank_prior
            + COVERAGE_WEIGHT * coverage
            + ANSWER_CUE_WEIGHT * answer_cue
        )
        reranked.append(
            {
                **candidate,
                "retrieval_score": candidate.get("score", 0.0),
                "score": rerank_score,
                "rerank_score": rerank_score,
                "query_coverage": coverage,
                "answer_cue_score": answer_cue,
            }
        )

    reranked.sort(
        key=lambda item: (
            -item["score"],
            -item["retrieval_score"],
        )
    )
    return reranked[:top_k]
