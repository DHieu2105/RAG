from unittest.mock import patch

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.pipeline import (
    _select_cited_sources,
    _select_sources,
    _validate_claim_coverage,
    chat,
)


def test_select_cited_sources_keeps_only_model_selected_clauses() -> None:
    results = [
        {
            "part": "2.1",
            "article": 11,
            "clauses": [1, 2, 3, 4],
            "page_start": 23,
            "page_end": 23,
            "text": "1. Cảnh báo.\n\n2. Ba lần cảnh báo liên tiếp thì bị buộc thôi học.\n\n3. Thông báo.\n\n4. Bảo lưu.",
        },
        {
            "article": 19,
            "clauses": [1],
            "page_start": 31,
            "page_end": 31,
        },
    ]

    sources = _select_cited_sources(
        results,
        [
            {
                "claim": "Ba lần cảnh báo liên tiếp thì bị buộc thôi học.",
                "document": 1,
                "clauses": [2],
                "evidence": "Ba lần cảnh báo liên tiếp thì bị buộc thôi học.",
            }
        ],
    )

    assert len(sources) == 1
    assert sources[0]["article"] == 11
    assert sources[0]["clauses"] == [2]
    assert sources[0]["claims"] == [
        "Ba lần cảnh báo liên tiếp thì bị buộc thôi học."
    ]
    assert sources[0]["evidence"] == [
        "Ba lần cảnh báo liên tiếp thì bị buộc thôi học."
    ]


def test_select_cited_sources_rejects_clause_outside_chunk() -> None:
    results = [
        {
            "article": 11,
            "clauses": [1, 2],
            "page_start": 23,
            "page_end": 23,
            "text": "1. Ngưỡng cảnh báo.\n\n2. Ba lần cảnh báo.",
        }
    ]

    try:
        _select_cited_sources(
            results,
            [
                {
                    "claim": "Một claim.",
                    "document": 1,
                    "clauses": [4],
                    "evidence": "Nội dung không tồn tại.",
                }
            ],
        )
    except ValueError as error:
        assert "không thuộc tài liệu" in str(error)
        return

    raise AssertionError("Expected ValueError to be raised")


def test_select_cited_sources_rejects_evidence_from_wrong_clause() -> None:
    results = [
        {
            "article": 11,
            "clauses": [1, 2],
            "page_start": 23,
            "page_end": 23,
            "text": (
                "1. Điểm dưới 0,8 ở học kỳ đầu thì bị cảnh báo.\n\n"
                "2. Ba lần cảnh báo liên tiếp thì bị buộc thôi học."
            ),
        }
    ]

    try:
        _select_cited_sources(
            results,
            [
                {
                    "claim": "Ba lần cảnh báo thì bị buộc thôi học.",
                    "document": 1,
                    "clauses": [2],
                    "evidence": "Điểm dưới 0,8 ở học kỳ đầu thì bị cảnh báo.",
                }
            ],
        )
    except ValueError as error:
        assert "Evidence không xuất hiện" in str(error)
        return

    raise AssertionError("Expected ValueError to be raised")


def test_claim_coverage_rejects_uncited_answer_sentence() -> None:
    answer = "Ngưỡng cảnh báo là 0,8. Ba lần cảnh báo thì bị thôi học."
    citations = [
        {
            "claim": "Ngưỡng cảnh báo là 0,8.",
            "document": 1,
            "clauses": [1],
        }
    ]

    try:
        _validate_claim_coverage(answer, citations)
    except ValueError as error:
        assert "chưa được gắn citation" in str(error)
        return

    raise AssertionError("Expected ValueError to be raised")


def test_select_sources_limits_and_deduplicates_sources() -> None:
    results = [
        {
            "part": "2.1",
            "article": 11,
            "clauses": [1],
            "page_start": 23,
            "page_end": 23,
        },
        {
            "part": "2.1",
            "article": 11,
            "clauses": [2],
            "page_start": 23,
            "page_end": 23,
        },
        {
            "part": "2.1",
            "article": 13,
            "clauses": [1],
            "page_start": 27,
            "page_end": 27,
        },
    ]

    sources = _select_sources(results)

    assert len(sources) == 2
    assert sources[0]["article"] == 11
    assert sources[0]["clauses"] == [1, 2]
    assert sources[1]["article"] == 13
    assert sources[0]["page_end"] == 23


def test_chat_builds_context_and_generates_answer() -> None:
    chunks = [
        {
            "article": 11,
            "clauses": [1],
            "page_start": 23,
            "text": "Sinh viên bị cảnh báo học tập.",
        }
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunks[0]["text"]])
    vectors = vectorizer.transform([chunks[0]["text"]])

    with patch(
        "app.rag.from_scratch.pipeline.generate_answer",
        return_value="Câu trả lời từ Ollama.",
    ) as mocked_generate_answer:
        answer = chat(
            "Khi nào sinh viên bị cảnh báo học tập?",
            chunks,
            vectorizer,
            vectors,
        )

    assert answer == "Câu trả lời từ Ollama."
    mocked_generate_answer.assert_called_once()
    question, context = mocked_generate_answer.call_args.args
    assert question == "Khi nào sinh viên bị cảnh báo học tập?"
    assert "[Tài liệu 1]" in context
    assert "Điều: 11" in context
    assert "Khoản: 1" in context
    assert "Trang PDF: 23" in context
    assert "Sinh viên bị cảnh báo học tập." in context


def test_chat_skips_ollama_when_nothing_is_retrieved() -> None:
    chunks = [
        {
            "article": 11,
            "clauses": [1],
            "page_start": 23,
            "text": "Sinh viên bị cảnh báo học tập.",
        }
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunks[0]["text"]])
    vectors = vectorizer.transform([chunks[0]["text"]])

    with patch(
        "app.rag.from_scratch.pipeline.generate_answer"
    ) as mocked_generate_answer:
        answer = chat(
            "một câu hỏi không có token phù hợp",
            chunks,
            vectorizer,
            vectors,
        )

    assert answer == "Không tìm thấy thông tin liên quan trong sổ tay."
    mocked_generate_answer.assert_not_called()


if __name__ == "__main__":
    test_chat_builds_context_and_generates_answer()
    test_chat_skips_ollama_when_nothing_is_retrieved()
    print("All tests passed!")
