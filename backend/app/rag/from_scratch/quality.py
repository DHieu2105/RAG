"""Answer-level regression metrics for verified RAG pipeline responses."""

from dataclasses import dataclass

from .pipeline import AnswerState, classify_answer_state
from .query_understanding import analyze_query, lexical_terms


@dataclass(frozen=True)
class AnswerQualityResult:
    citation_correctness: bool
    faithfulness: bool
    completeness: bool
    scope_accuracy: bool
    abstention_accuracy: bool


def _source_text(source: dict) -> str:
    values = [
        str(source.get("article_title") or ""),
        *(str(value) for value in source.get("claims", [])),
        *(str(value) for value in source.get("evidence", [])),
    ]
    return " ".join(values)


def evaluate_answer_response(
    question: str,
    *,
    answerable: bool,
    response: dict,
) -> AnswerQualityResult:
    """Score a response using guarantees enforced by the production validator.

    A response is marked verified only after exact evidence, clause, scope,
    numeric/negation, and extracted-unit coverage checks have passed. This
    evaluator exposes those guarantees as regression metrics without keeping a
    second set of handbook answers in code.
    """
    answer = str(response.get("answer") or "")
    sources = response.get("sources")
    sources = sources if isinstance(sources, list) else []
    state = classify_answer_state(answer)
    verified = state == AnswerState.VERIFIED

    citation_correct = verified and bool(sources) and all(
        source.get("claims") and source.get("evidence") for source in sources
    )
    source_terms = set(
        lexical_terms(" ".join(_source_text(source) for source in sources))
    )
    focus = set(analyze_query(question).focus_terms)
    scope_accuracy = citation_correct and (
        not focus or len(focus & source_terms) / len(focus) >= 0.75
    )

    if answerable:
        abstention_accuracy = state != AnswerState.NO_RELEVANT_DOCUMENT
        faithfulness = citation_correct
        completeness = verified
    else:
        abstention_accuracy = state == AnswerState.NO_RELEVANT_DOCUMENT
        faithfulness = not verified
        completeness = abstention_accuracy
        scope_accuracy = abstention_accuracy
        citation_correct = not sources

    return AnswerQualityResult(
        citation_correctness=citation_correct,
        faithfulness=faithfulness,
        completeness=completeness,
        scope_accuracy=scope_accuracy,
        abstention_accuracy=abstention_accuracy,
    )
