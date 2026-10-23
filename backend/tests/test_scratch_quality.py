from app.rag.from_scratch.pipeline import (
    INVALID_CITATION_ANSWER,
    NO_RELEVANT_DOCUMENT_ANSWER,
)
from app.rag.from_scratch.quality import evaluate_answer_response


def test_quality_metrics_accept_a_verified_grounded_response() -> None:
    response = {
        "answer": "Mỗi học kỳ chính đăng ký tối thiểu 12 tín chỉ.",
        "sources": [
            {
                "article_title": "Điều 7. Tổ chức đăng ký học tập",
                "claims": ["Mỗi học kỳ chính đăng ký tối thiểu 12 tín chỉ."],
                "evidence": ["tối thiểu 12 tín chỉ"],
            }
        ],
    }

    metrics = evaluate_answer_response(
        "Mỗi học kỳ chính đăng ký tối thiểu bao nhiêu tín chỉ?",
        answerable=True,
        response=response,
    )

    assert metrics.citation_correctness
    assert metrics.faithfulness
    assert metrics.completeness
    assert metrics.scope_accuracy
    assert metrics.abstention_accuracy


def test_quality_metrics_accept_correct_out_of_scope_abstention() -> None:
    metrics = evaluate_answer_response(
        "Trường có cấp laptop miễn phí không?",
        answerable=False,
        response={"answer": NO_RELEVANT_DOCUMENT_ANSWER, "sources": []},
    )

    assert metrics.citation_correctness
    assert metrics.faithfulness
    assert metrics.completeness
    assert metrics.scope_accuracy
    assert metrics.abstention_accuracy


def test_quality_metrics_do_not_treat_invalid_citation_as_verified() -> None:
    metrics = evaluate_answer_response(
        "Điều kiện tốt nghiệp là gì?",
        answerable=True,
        response={"answer": INVALID_CITATION_ANSWER, "sources": []},
    )

    assert not metrics.citation_correctness
    assert not metrics.faithfulness
    assert not metrics.completeness
