from app.rag.from_scratch.reranker import (
    lexical_terms,
    rerank_results,
)


def _candidate(text: str, score: float) -> dict:
    return {
        "part": "2.1",
        "article": 11,
        "clauses": [1],
        "page_start": 23,
        "text": text,
        "score": score,
    }


def test_lexical_terms_removes_generic_question_words() -> None:
    terms = lexical_terms("Khi nào sinh viên bị buộc thôi học?")

    assert terms == {"buộc", "thôi", "học"}


def test_reranker_prefers_direct_conditions_over_post_event_content() -> None:
    post_event = _candidate(
        "Sinh viên bị buộc thôi học nếu có nhu cầu được xét chuyển hệ.",
        0.05,
    )
    direct_answer = _candidate(
        "Sinh viên bị buộc thôi học trong các trường hợp sau: "
        "thuộc diện cảnh báo kết quả học tập ba lần.",
        0.049,
    )
    queries = [
        "Khi nào sinh viên bị buộc thôi học?",
        "buộc thôi học do kết quả học tập yếu",
        "cảnh báo kết quả học tập",
    ]

    results = rerank_results(
        queries[0],
        [post_event, direct_answer],
        queries,
    )

    assert results[0]["text"] == direct_answer["text"]
    assert results[0]["answer_cue_score"] > 0.0
    assert results[0]["retrieval_score"] == direct_answer["score"]


def test_reranker_is_stable_for_equal_features() -> None:
    first = _candidate("nội dung giống nhau", 0.2)
    second = _candidate("nội dung giống nhau", 0.1)

    results = rerank_results("câu hỏi", [first, second], ["câu hỏi"])

    assert results[0]["retrieval_score"] == 0.2


def test_reranker_validates_top_k() -> None:
    try:
        rerank_results("query", [], ["query"], top_k=0)
    except ValueError:
        return

    raise AssertionError("Expected invalid top_k to raise ValueError")
