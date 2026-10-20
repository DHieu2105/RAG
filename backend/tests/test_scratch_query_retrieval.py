from unittest.mock import patch

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.pipeline import (
    ask,
    ask_with_query_processing,
    chat,
)
from app.rag.from_scratch.retriever import reciprocal_rank_fusion


def _chunk(part: str, text: str, score: float) -> dict:
    return {
        "part": part,
        "article": 3,
        "clauses": [1],
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "score": score,
    }


def test_fusion_does_not_merge_equal_articles_from_different_parts() -> None:
    training = _chunk("2.1", "Điều 3 quy chế đào tạo", 0.9)
    tuition = _chunk("2.6", "Điều 3 quy định học phí", 0.8)

    results = reciprocal_rank_fusion(
        [[training, tuition], [tuition, training]],
        top_k=5,
    )

    assert len(results) == 2
    assert {result["part"] for result in results} == {"2.1", "2.6"}


def test_fusion_deduplicates_chunks_and_is_stable() -> None:
    first = _chunk("2.1", "Nội dung thứ nhất", 0.9)
    second = _chunk("2.1", "Nội dung thứ hai", 0.8)
    ranked_results = [[first, second], [second, first]]

    first_run = reciprocal_rank_fusion(ranked_results, top_k=2)
    second_run = reciprocal_rank_fusion(ranked_results, top_k=2)

    assert first_run == second_run
    assert [result["text"] for result in first_run] == [
        "Nội dung thứ nhất",
        "Nội dung thứ hai",
    ]
    assert all(result["matched_queries"] == 2 for result in first_run)


def test_question_without_expansion_matches_baseline_retrieval() -> None:
    chunks = [
        {"text": "sinh viên đăng ký tín chỉ tối đa", "article": 7},
        {"text": "sinh viên được xét học bổng", "article": None},
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunk["text"] for chunk in chunks])
    vectors = vectorizer.transform([chunk["text"] for chunk in chunks])
    question = "Sinh viên đăng ký tối đa bao nhiêu tín chỉ?"

    baseline = ask(question, chunks, vectorizer, vectors)
    expanded = ask_with_query_processing(
        question,
        chunks,
        vectorizer,
        vectors,
    )

    assert expanded == baseline


def test_chat_keeps_original_question_when_search_is_expanded() -> None:
    chunks = [
        {
            "text": "buộc thôi học do kết quả học tập yếu",
            "article": 11,
            "clauses": [1, 2],
            "page_start": 23,
        },
        {
            "text": "điểm đánh giá học phần thấp",
            "article": 9,
            "clauses": [3],
            "page_start": 19,
        },
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunk["text"] for chunk in chunks])
    vectors = vectorizer.transform([chunk["text"] for chunk in chunks])
    question = "Điểm thấp quá thì có bị đuổi học không?"

    with patch(
        "app.rag.from_scratch.pipeline.generate_answer",
        return_value="Câu trả lời.",
    ) as mocked_generate_answer:
        chat(question, chunks, vectorizer, vectors)

    ollama_question = mocked_generate_answer.call_args.args[0]
    assert ollama_question == question


def test_fusion_validates_configuration() -> None:
    try:
        reciprocal_rank_fusion([], top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid top_k to raise ValueError")

    try:
        reciprocal_rank_fusion([[]], weights=[])
    except ValueError:
        return

    raise AssertionError("Expected mismatched weights to raise ValueError")
