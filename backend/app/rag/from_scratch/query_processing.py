"""Deterministic query expansion for Vietnamese academic terminology."""

import re
import unicodedata

from .query_understanding import QueryIntent, analyze_query, generation_requirements


DEFAULT_MAX_QUERIES = 5


def normalize_query(query: str) -> str:
    """Normalize Unicode and collapse whitespace without changing meaning."""
    normalized = unicodedata.normalize("NFC", query)
    return " ".join(normalized.split())


def _contains_phrase(query: str, phrase: str) -> bool:
    return phrase.casefold() in query.casefold()


def is_complete_list_question(question: str) -> bool:
    """Return whether the user requests a generic exhaustive enumeration."""
    return analyze_query(question).has_intent(QueryIntent.ENUMERATION)


def generate_answer_requirements(question: str) -> list[str]:
    """Return generic intent requirements without handbook answer facts."""
    return generation_requirements(analyze_query(question))


def _replace_phrase(query: str, phrase: str, replacement: str) -> str:
    return re.sub(
        re.escape(phrase),
        replacement,
        query,
        flags=re.IGNORECASE,
    )


def _append_unique(queries: list[str], candidate: str) -> None:
    normalized = normalize_query(candidate)
    if not normalized:
        return

    existing = {query.casefold() for query in queries}
    if normalized.casefold() not in existing:
        queries.append(normalized)


# Vocabulary-only rewrites; these never add answer facts or neighboring topics.
_SEARCH_TERM_REPLACEMENTS = (
    (("đuổi học", "đuổi khỏi trường"), "buộc thôi học"),
    (("ra trường",), "tốt nghiệp"),
    (("đăng ký tín chỉ", "đăng ký học phần"), "đăng ký học"),
    (("học lại",), "học phần không đạt"),
    (("điểm thấp", "điểm kém", "học lực yếu"), "kết quả học tập yếu"),
    (("miễn giảm học phí",), "miễn, giảm học phí"),
)


def generate_search_queries(
    question: str,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[str]:
    """Keep the original query and add a bounded set of search-only variants."""
    if max_queries <= 0:
        raise ValueError("max_queries must be greater than 0")

    original = normalize_query(question)
    if not original:
        return []

    queries = [original]
    rewritten = original

    for aliases, canonical in _SEARCH_TERM_REPLACEMENTS:
        if canonical == "kết quả học tập yếu" and not any(
            context in original.casefold()
            for context in ("học", "sinh viên", "môn", "thi", "tín chỉ")
        ):
            continue
        for alias in aliases:
            if _contains_phrase(rewritten, alias):
                rewritten = _replace_phrase(
                    rewritten,
                    alias,
                    canonical,
                )
                break

    _append_unique(queries, rewritten)

    analysis = analyze_query(original)
    if analysis.has_intent(QueryIntent.PROCEDURE):
        procedure_scope = rewritten
        for marker in (
            "quy trình", "thủ tục", "các bước", "làm thế nào",
            "như thế nào", "cách thực hiện", "thực hiện",
        ):
            procedure_scope = _replace_phrase(procedure_scope, marker, "")
        procedure_scope = procedure_scope.strip(" ?.!,")
        _append_unique(queries, f"hướng dẫn {procedure_scope}")
    elif analysis.has_intent(QueryIntent.CONDITIONS):
        condition_scope = rewritten
        for marker in ("khi nào", "điều kiện", "là gì"):
            condition_scope = _replace_phrase(condition_scope, marker, "")
        condition_scope = condition_scope.strip(" ?.!,")
        _append_unique(queries, f"điều kiện {condition_scope}")
    return queries[:max_queries]
