from unittest.mock import patch

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.pipeline import (
    INCOMPLETE_CONTEXT_ANSWER,
    INVALID_CITATION_ANSWER,
    _add_title_matches,
    _expand_same_article_context,
    _select_cited_sources,
    _select_sources,
    _validate_claim_coverage,
    _validate_required_coverage,
    chat,
    chat_with_sources,
)
from app.rag.from_scratch.query_understanding import analyze_query


def _tuition_results() -> list[dict]:
    return [
        {
            "part": "2.6",
            "article": 3,
            "article_title": "Điều 3. Đối tượng được miễn, giảm học phí",
            "clauses": [1],
            "page_start": 47,
            "page_end": 48,
            "text": "Các đối tượng được miễn, giảm học phí.",
        },
        {
            "part": "2.6",
            "article": 4,
            "article_title": "Điều 4. Trợ cấp xã hội",
            "clauses": [1],
            "page_start": 49,
            "page_end": 49,
            "text": "Các đối tượng được hưởng trợ cấp xã hội.",
        },
    ]


def test_title_match_rescues_relevant_article_missed_by_content_ranking() -> None:
    question = "Ai được miễn học phí?"
    wrong = {
        "part": "2.1",
        "article": 7,
        "article_title": "Điều 7. Tổ chức đăng ký học tập",
        "clauses": [5],
        "page_start": 20,
        "text": "Sinh viên được miễn học phần đã được công nhận tín chỉ.",
    }
    target = {
        "part": "2.6",
        "article": 3,
        "article_title": "Điều 3. Các đối tượng được miễn, giảm học phí",
        "clauses": [1],
        "page_start": 47,
        "text": "1. Nhóm Alpha được miễn 100% học phí.",
    }

    augmented = _add_title_matches(
        analyze_query(question),
        [wrong],
        [wrong, target],
    )
    selected = _expand_same_article_context(
        question,
        augmented,
        [wrong, target],
    )

    assert target in augmented
    assert [chunk["article"] for chunk in selected] == [3]
    assert [chunk["clauses"] for chunk in selected] == [[1]]


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


def test_select_cited_sources_accepts_fragment_of_one_oversized_clause() -> None:
    results = [
        {
            "article": 3,
            "clauses": [1],
            "page_start": 48,
            "page_end": 48,
            "text": "đoạn tiếp theo của khoản một về nhóm được miễn học phí",
        }
    ]

    sources = _select_cited_sources(
        results,
        [
            {
                "claim": "Nhóm này được miễn học phí.",
                "document": 1,
                "clauses": [1],
                "evidence": "nhóm được miễn học phí",
            }
        ],
    )

    assert sources[0]["article"] == 3
    assert sources[0]["clauses"] == [1]


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


def test_complete_list_retrieval_keeps_only_relevant_anchor_scope() -> None:
    article_3_first = {
        "part": "2.6",
        "article": 3,
        "article_title": "Điều 3. Miễn, giảm học phí",
        "clauses": [1],
        "page_start": 47,
        "text": "Nhóm miễn học phí thứ nhất.",
    }
    article_3_second = {
        "part": "2.6",
        "article": 3,
        "article_title": "Điều 3. Miễn, giảm học phí",
        "clauses": [1, 2, 3],
        "page_start": 48,
        "text": "Các nhóm miễn còn lại và các mức giảm học phí.",
    }
    article_4 = {
        "part": "2.6",
        "article": 4,
        "article_title": "Điều 4. Trợ cấp xã hội",
        "clauses": [1],
        "page_start": 49,
        "text": "Trợ cấp xã hội.",
    }

    expanded = _expand_same_article_context(
        "Đối tượng nào được miễn, giảm học phí?",
        [article_3_second, article_4],
        [article_3_first, article_3_second, article_4],
    )

    assert [chunk["text"] for chunk in expanded] == [article_3_second["text"]]
    assert all(chunk["article"] == 3 for chunk in expanded)


def _dynamic_list_results() -> list[dict]:
    return [
        {
            "part": "2.9",
            "article": 8,
            "article_title": "Điều 8. Danh sách quyền lợi thử nghiệm",
            "clauses": [1],
            "page_start": 80,
            "text": "1. Các quyền lợi gồm: a. Quyền lợi Alpha. b. Quyền lợi Beta. c. Quyền lợi Gamma.",
        }
    ]


def _dynamic_citations() -> list[dict]:
    return [
        {
            "claim": f"Sinh viên có quyền lợi {name}.",
            "document": 1,
            "clauses": [1],
            "evidence": f"{label}. Quyền lợi {name}.",
        }
        for label, name in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"))
    ]


def test_coverage_accepts_all_units_extracted_from_document() -> None:
    citations = _dynamic_citations()
    _validate_required_coverage(
        "Liệt kê các quyền lợi thử nghiệm.",
        {"answer": "\n".join(item["claim"] for item in citations), "citations": citations},
        _dynamic_list_results(),
    )


def test_coverage_has_no_fixed_topic_count() -> None:
    results = _dynamic_list_results()
    results[0]["text"] = "1. Các quyền lợi gồm: a. Quyền lợi Alpha. b. Quyền lợi Beta."
    citations = _dynamic_citations()[:2]
    _validate_required_coverage(
        "Liệt kê các quyền lợi thử nghiệm.",
        {"answer": "\n".join(item["claim"] for item in citations), "citations": citations},
        results,
    )


def test_coverage_rejects_an_extracted_unit_missing_from_answer() -> None:
    citations = _dynamic_citations()[:2]
    try:
        _validate_required_coverage(
            "Liệt kê các quyền lợi thử nghiệm.",
            {"answer": "\n".join(item["claim"] for item in citations), "citations": citations},
            _dynamic_list_results(),
        )
    except ValueError as error:
        assert "thiếu các đơn vị evidence" in str(error)
        return
    raise AssertionError("Expected ValueError to be raised")


def test_coverage_rejects_evidence_bound_to_wrong_document() -> None:
    results = _dynamic_list_results() + [
        {"part": "2.9", "article": 9, "clauses": [1], "text": "Nội dung liền kề."}
    ]
    citations = _dynamic_citations()
    citations[-1]["document"] = 2
    try:
        _validate_required_coverage(
            "Liệt kê các quyền lợi thử nghiệm.",
            {"answer": "\n".join(item["claim"] for item in citations), "citations": citations},
            results,
        )
    except ValueError as error:
        assert "thiếu các đơn vị evidence" in str(error)
        return
    raise AssertionError("Expected ValueError to be raised")


def test_chat_retries_once_when_tuition_answer_is_incomplete() -> None:
    results = [
        {
            **_tuition_results()[0],
            "text": (
                "1. Đối tượng được miễn học phí gồm: "
                "a. Sinh viên nhóm Alpha. b. Sinh viên nhóm Beta."
            ),
        }
    ]
    incomplete = {
        "answer": "Sinh viên nhóm Alpha được miễn học phí.",
        "citations": [
            {
                "claim": "Sinh viên nhóm Alpha được miễn học phí.",
                "document": 1,
                "clauses": [1],
                "evidence": "a. Sinh viên nhóm Alpha.",
            }
        ],
    }
    complete_citations = [
        *incomplete["citations"],
        {
            "claim": "Sinh viên nhóm Beta được miễn học phí.",
            "document": 1,
            "clauses": [1],
            "evidence": "b. Sinh viên nhóm Beta.",
        },
    ]
    complete = {
        "answer": "\n\n".join(
            citation["claim"] for citation in complete_citations
        ),
        "citations": complete_citations,
    }

    with (
        patch(
            "app.rag.from_scratch.pipeline.retrieve_for_chat",
            return_value=results,
        ),
        patch(
            "app.rag.from_scratch.pipeline.generate_grounded_answer",
            side_effect=[incomplete, complete],
        ) as mocked_generate,
    ):
        response = chat_with_sources(
            "Đối tượng nào được miễn, giảm học phí?",
            results,
            object(),
            [],
        )

    assert response["answer"] == complete["answer"]
    assert mocked_generate.call_count == 2
    retry_requirements = mocked_generate.call_args.kwargs[
        "answer_requirements"
    ]
    assert any(
        "Bản trả lời trước không đạt kiểm tra" in item
        for item in retry_requirements
    )


def test_chat_returns_insufficient_answer_after_two_invalid_attempts() -> None:
    results = [
        {
            **_tuition_results()[0],
            "text": (
                "1. Đối tượng được miễn học phí gồm: "
                "a. Sinh viên nhóm Alpha. b. Sinh viên nhóm Beta."
            ),
        }
    ]
    incomplete = {
        "answer": "Sinh viên nhóm Alpha được miễn học phí.",
        "citations": [
            {
                "claim": "Sinh viên nhóm Alpha được miễn học phí.",
                "document": 1,
                "clauses": [1],
                "evidence": "a. Sinh viên nhóm Alpha.",
            }
        ],
    }

    with (
        patch(
            "app.rag.from_scratch.pipeline.retrieve_for_chat",
            return_value=results,
        ),
        patch(
            "app.rag.from_scratch.pipeline.generate_grounded_answer",
            side_effect=[incomplete, incomplete],
        ),
    ):
        response = chat_with_sources(
            "Đối tượng nào được miễn, giảm học phí?",
            results,
            object(),
            [],
        )

    assert response == {
        "answer": "Đã tìm thấy căn cứ liên quan nhưng câu trả lời sinh ra chưa bao phủ đầy đủ thông tin.",
        "sources": [],
    }


def test_chat_distinguishes_incomplete_retrieval_context() -> None:
    results = [
        {
            "article": 8,
            "clauses": [1],
            "text": "Đối tượng Alpha.",
            "_context_complete": False,
        }
    ]
    with (
        patch(
            "app.rag.from_scratch.pipeline.retrieve_for_chat",
            return_value=results,
        ),
        patch(
            "app.rag.from_scratch.pipeline.generate_grounded_answer"
        ) as mocked_generate,
    ):
        response = chat_with_sources(
            "Những đối tượng Alpha nào?",
            results,
            object(),
            [],
        )

    assert response == {"answer": INCOMPLETE_CONTEXT_ANSWER, "sources": []}
    mocked_generate.assert_not_called()


def test_chat_distinguishes_invalid_citation_from_missing_context() -> None:
    results = [
        {
            "article": 8,
            "clauses": [1],
            "text": "Mức Alpha là 12 đơn vị.",
        }
    ]
    invalid = {
        "answer": "Mức Alpha là 12 đơn vị.",
        "citations": [
            {
                "claim": "Mức Alpha là 12 đơn vị.",
                "document": 1,
                "clauses": [1],
                "evidence": "Đoạn không tồn tại.",
            }
        ],
    }
    with (
        patch(
            "app.rag.from_scratch.pipeline.retrieve_for_chat",
            return_value=results,
        ),
        patch(
            "app.rag.from_scratch.pipeline.generate_grounded_answer",
            side_effect=[invalid, invalid],
        ),
    ):
        response = chat_with_sources(
            "Mức Alpha là gì?",
            results,
            object(),
            [],
        )

    assert response == {"answer": INVALID_CITATION_ANSWER, "sources": []}


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

    assert answer == "Không tìm thấy tài liệu liên quan trong sổ tay."
    mocked_generate_answer.assert_not_called()


if __name__ == "__main__":
    test_chat_builds_context_and_generates_answer()
    test_chat_skips_ollama_when_nothing_is_retrieved()
    print("All tests passed!")
