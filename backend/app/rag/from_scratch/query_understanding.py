"""Generic Vietnamese query understanding for retrieval and generation."""

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata


class QueryIntent(str, Enum):
    """Question shapes that affect retrieval and answer construction."""

    FACTUAL_LOOKUP = "factual_lookup"
    ENUMERATION = "enumeration"
    CONDITIONS = "conditions"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    NUMERIC_THRESHOLD = "numeric_threshold"
    SCENARIO_ELIGIBILITY = "scenario_eligibility"


@dataclass(frozen=True)
class QueryAnalysis:
    """A topic-independent description of a user question."""

    normalized_question: str
    intents: tuple[QueryIntent, ...]
    focus_terms: tuple[str, ...]
    requires_complete_coverage: bool

    def has_intent(self, intent: QueryIntent) -> bool:
        return intent in self.intents


_INTENT_MARKERS = {
    QueryIntent.ENUMERATION: (
        "liệt kê", "đầy đủ", "tất cả", "toàn bộ", "đối tượng nào",
        "những đối tượng", "các đối tượng", "trường hợp nào",
        "những trường hợp", "các trường hợp", "ai được", "ai sẽ được",
        "những ai", "gồm những ai", "bao gồm những ai", "những nhóm nào",
        "có những gì",
    ),
    QueryIntent.CONDITIONS: (
        "điều kiện", "khi nào", "trường hợp nào", "có được", "có bị",
        "đủ điều kiện", "cần đáp ứng", "cần những gì",
    ),
    QueryIntent.COMPARISON: (
        "so sánh", "khác nhau", "phân biệt", "khác gì", "hay là",
        " so với ", " vs ",
    ),
    QueryIntent.PROCEDURE: (
        "quy trình", "thủ tục", "các bước", "làm thế nào", "như thế nào",
        "cách thực hiện", "hướng dẫn",
    ),
    QueryIntent.NUMERIC_THRESHOLD: (
        "bao nhiêu", "mấy ", "tối thiểu", "tối đa", "ngưỡng", "mức ",
        "dưới ", "trên ", "ít nhất", "nhiều nhất", "%",
    ),
}
_SCENARIO_PRONOUNS = ("em ", "tôi ", "mình ", "trường hợp của em")
_SCENARIO_DECISIONS = (
    "có được", "được không", "có bị", "có phải", "đủ điều kiện",
    "thuộc diện",
)
_FOCUS_STOP_WORDS = {
    "ai", "bao", "các", "cần", "cho", "có", "của", "để", "điều",
    "được", "em", "gì", "hay", "hiện", "khi", "không", "là", "làm",
    "bị", "chính", "học", "hướng", "ít", "kỳ", "mình", "mấy", "mỗi",
    "nào", "nhà", "nhất", "nhiêu", "như",
    "những", "phải", "quy", "ra", "sao", "sinh", "sổ", "tay", "thế",
    "thì", "thủ", "thực", "tôi", "trình", "trong", "trường", "tự",
    "tục", "viên", "và", "về", "với", "bước", "dẫn",
}


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFC", query)
    return " ".join(normalized.split())


def lexical_terms(text: str) -> tuple[str, ...]:
    """Return stable Unicode word tokens for scope and evidence checks."""
    return tuple(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    folded = f" {text.casefold()} "
    return any(marker.casefold() in folded for marker in markers)


def _focus_terms(question: str) -> tuple[str, ...]:
    ordered: list[str] = []
    for term in lexical_terms(question):
        if len(term) <= 1 or term in _FOCUS_STOP_WORDS or term.isdigit():
            continue
        if term not in ordered:
            ordered.append(term)
    return tuple(ordered)


def analyze_query(question: str) -> QueryAnalysis:
    """Infer generic answer shape and evidence scope from a question."""
    normalized = normalize_query(question)
    intents: list[QueryIntent] = [QueryIntent.FACTUAL_LOOKUP]
    for intent, markers in _INTENT_MARKERS.items():
        if _contains_any(normalized, markers):
            intents.append(intent)

    folded = f" {normalized.casefold()} "
    if any(pronoun in folded for pronoun in _SCENARIO_PRONOUNS) and any(
        decision in folded for decision in _SCENARIO_DECISIONS
    ):
        intents.append(QueryIntent.SCENARIO_ELIGIBILITY)

    complete = any(
        intent in intents
        for intent in (
            QueryIntent.ENUMERATION,
            QueryIntent.CONDITIONS,
            QueryIntent.PROCEDURE,
        )
    )
    return QueryAnalysis(
        normalized_question=normalized,
        intents=tuple(intents),
        focus_terms=_focus_terms(normalized),
        requires_complete_coverage=complete,
    )


def generation_requirements(analysis: QueryAnalysis) -> list[str]:
    """Build intent-level instructions without embedding handbook answers."""
    requirements: list[str] = []
    if analysis.has_intent(QueryIntent.ENUMERATION):
        requirements.append(
            "Liệt kê đủ từng đơn vị trong checklist coverage trích từ context; "
            "không thêm mục ngoài phạm vi câu hỏi."
        )
    if analysis.has_intent(QueryIntent.CONDITIONS):
        requirements.append(
            "Giữ đủ điều kiện, quan hệ và/hoặc, ngưỡng và ngoại lệ trực tiếp "
            "quyết định kết luận được hỏi."
        )
    if analysis.has_intent(QueryIntent.COMPARISON):
        requirements.append(
            "So sánh đúng các khái niệm được hỏi bằng căn cứ riêng; không đồng "
            "nhất các chế độ gần nhau."
        )
    if analysis.has_intent(QueryIntent.PROCEDURE):
        requirements.append(
            "Trình bày đúng thứ tự các bước trong context; chỉ thêm điều kiện "
            "hoặc ngoại lệ gắn trực tiếp với quy trình."
        )
    if analysis.has_intent(QueryIntent.NUMERIC_THRESHOLD):
        requirements.append(
            "Giữ nguyên con số, đơn vị, dấu so sánh và ngoại lệ đi kèm."
        )
    if analysis.has_intent(QueryIntent.SCENARIO_ELIGIBILITY):
        requirements.append(
            "Chỉ kết luận đủ/không đủ điều kiện khi tình huống khớp đủ quy "
            "định; nếu thiếu dữ kiện phải nêu dữ kiện còn thiếu."
        )
    return requirements or ["Trả lời trực tiếp, ngắn gọn, đúng phạm vi câu hỏi."]
