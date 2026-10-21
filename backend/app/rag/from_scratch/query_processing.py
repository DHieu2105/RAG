"""Deterministic query expansion for Vietnamese academic terminology."""

from dataclasses import dataclass
import re
import unicodedata


DEFAULT_MAX_QUERIES = 4


@dataclass(frozen=True)
class QueryExpansionRule:
    """Map informal academic phrases to a canonical handbook term."""

    name: str
    aliases: tuple[str, ...]
    canonical: str
    context_any: tuple[str, ...]
    related_queries: tuple[str, ...] = ()


TERM_RULES = (
    QueryExpansionRule(
        name="course_registration",
        aliases=("đăng ký tín chỉ", "đăng ký học phần"),
        canonical="đăng ký học",
        context_any=("đăng ký", "tín chỉ", "học phần", "học"),
        related_queries=(
            "hướng dẫn đăng ký học",
            "khối lượng học tập đăng ký trong học kỳ chính",
        ),
    ),
    QueryExpansionRule(
        name="academic_dismissal",
        aliases=("đuổi học", "đuổi khỏi trường"),
        canonical="buộc thôi học",
        context_any=("học", "sinh viên", "điểm", "trường", "kết quả"),
        related_queries=(
            "buộc thôi học do kết quả học tập yếu",
            "cảnh báo kết quả học tập",
        ),
    ),
    QueryExpansionRule(
        name="weak_academic_results",
        aliases=("điểm thấp", "điểm kém", "học lực yếu"),
        canonical="kết quả học tập yếu",
        context_any=(
            "học",
            "sinh viên",
            "môn",
            "thi",
            "tín chỉ",
            "cảnh báo",
            "buộc thôi",
            "đuổi",
        ),
    ),
    QueryExpansionRule(
        name="graduation",
        aliases=("ra trường",),
        canonical="tốt nghiệp",
        context_any=("học", "sinh viên", "bằng", "điều kiện", "chương trình"),
    ),
    QueryExpansionRule(
        name="failed_course",
        aliases=("học lại",),
        canonical="học phần không đạt",
        context_any=("học", "sinh viên", "môn", "thi", "điểm", "tín chỉ"),
    ),
)


def normalize_query(query: str) -> str:
    """Normalize Unicode and collapse whitespace without changing meaning."""
    normalized = unicodedata.normalize("NFC", query)
    return " ".join(normalized.split())


def _contains_phrase(query: str, phrase: str) -> bool:
    return phrase.casefold() in query.casefold()


def _has_context(query: str, context_any: tuple[str, ...]) -> bool:
    folded_query = query.casefold()
    return any(term.casefold() in folded_query for term in context_any)


def matching_rules(question: str) -> list[QueryExpansionRule]:
    """Return expansion rules that match both a phrase and its context."""
    query = normalize_query(question)
    matches: list[QueryExpansionRule] = []

    for rule in TERM_RULES:
        terms = (*rule.aliases, rule.canonical)
        contains_term = any(_contains_phrase(query, term) for term in terms)
        if contains_term and _has_context(query, rule.context_any):
            matches.append(rule)

    return matches


def generate_answer_requirements(question: str) -> list[str]:
    """Return intent-level coverage checks without supplying answer facts."""
    rule_names = {rule.name for rule in matching_rules(question)}
    requirements: list[str] = []

    if "course_registration" in rule_names:
        requirements.extend(
            (
                "Tạo đúng một claim nêu cách đăng ký từ đúng mục hướng dẫn "
                "đăng ký học.",
                "Tạo đúng một claim nêu giới hạn tín chỉ của học kỳ chính và "
                "ngoại lệ gắn trực tiếp nếu tài liệu có.",
                "Không tạo claim khác về thời điểm chi tiết, hủy hoặc rút "
                "học phần khi câu hỏi không hỏi.",
            )
        )

    if {
        "academic_dismissal",
        "weak_academic_results",
    }.issubset(rule_names):
        requirements.extend(
            (
                "Mở đầu bằng việc trả lời điểm thấp một lần có trực tiếp dẫn "
                "tới buộc thôi học hay không.",
                "Nêu riêng các ngưỡng điểm dẫn tới cảnh báo học tập.",
                "Nêu riêng số lần cảnh báo liên tiếp dẫn tới buộc thôi học.",
            )
        )

    return requirements


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

    rules = matching_rules(original)
    queries = [original]
    rewritten = original

    for rule in rules:
        for alias in rule.aliases:
            if _contains_phrase(rewritten, alias):
                rewritten = _replace_phrase(
                    rewritten,
                    alias,
                    rule.canonical,
                )

    _append_unique(queries, rewritten)

    for rule in rules:
        for related_query in rule.related_queries:
            _append_unique(queries, related_query)
            if len(queries) >= max_queries:
                return queries[:max_queries]

    return queries[:max_queries]
