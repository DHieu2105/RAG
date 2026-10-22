import unicodedata

from app.rag.from_scratch.query_processing import (
    generate_answer_requirements,
    generate_search_queries,
    is_complete_list_question,
    is_tuition_policy_question,
    matching_rules,
    normalize_query,
)


def test_normalize_query_normalizes_unicode_and_whitespace() -> None:
    decomposed = unicodedata.normalize(
        "NFD",
        "  Điểm   thấp\n có bị đuổi học không?  ",
    )

    normalized = normalize_query(decomposed)

    assert normalized == "Điểm thấp có bị đuổi học không?"
    assert unicodedata.is_normalized("NFC", normalized)


def test_detects_contextual_academic_terms() -> None:
    rules = matching_rules("Điểm thấp quá thì có bị đuổi học không?")

    assert [rule.name for rule in rules] == [
        "academic_dismissal",
        "weak_academic_results",
    ]


def test_generates_bounded_search_queries_and_keeps_original() -> None:
    original = "Điểm thấp quá thì có bị đuổi học không?"

    queries = generate_search_queries(original)

    assert queries[0] == original
    assert len(queries) <= 4
    assert any(
        "kết quả học tập yếu" in query
        and "buộc thôi học" in query
        for query in queries
    )
    assert "buộc thôi học do kết quả học tập yếu" in queries
    assert "cảnh báo kết quả học tập" in queries


def test_expands_registration_into_rules_and_instructions() -> None:
    queries = generate_search_queries("Đăng ký tín chỉ")

    assert queries[0] == "Đăng ký tín chỉ"
    assert any("hướng dẫn đăng ký học" == query for query in queries)
    assert any("khối lượng học tập" in query for query in queries)


def test_registration_answer_requirements_cover_separate_claims() -> None:
    requirements = generate_answer_requirements("Đăng ký tín chỉ")

    assert any("mục hướng dẫn đăng ký học" in item for item in requirements)
    assert any("giới hạn tín chỉ" in item for item in requirements)
    assert any("hủy hoặc rút học phần" in item for item in requirements)


def test_low_score_answer_requirements_preserve_warning_chain() -> None:
    requirements = generate_answer_requirements(
        "Điểm thấp quá có bị đuổi học không?"
    )

    assert any("điểm thấp một lần" in item for item in requirements)
    assert any("ngưỡng điểm" in item for item in requirements)
    assert any("số lần cảnh báo" in item for item in requirements)


def test_tuition_question_expands_all_three_benefit_levels() -> None:
    question = "Đối tượng nào được miễn, giảm học phí?"

    queries = generate_search_queries(question)

    assert queries[0] == question
    assert "đối tượng được miễn 100% học phí không phải đóng học phí" in queries
    assert "đối tượng được giảm 70% học phí" in queries
    assert "đối tượng được giảm 50% học phí" in queries
    assert is_complete_list_question(question)
    assert is_tuition_policy_question(question)


def test_tuition_answer_requirements_demand_complete_separate_groups() -> None:
    requirements = generate_answer_requirements(
        "Những đối tượng sinh viên nào được miễn học phí?"
    )

    assert any("sáu đối tượng" in item and "100%" in item for item in requirements)
    assert any("70%" in item for item in requirements)
    assert any("50%" in item for item in requirements)
    assert any("trợ cấp xã hội" in item for item in requirements)


def test_personal_tuition_question_does_not_require_full_policy_list() -> None:
    question = "Em có được miễn học phí không?"

    assert is_tuition_policy_question(question)
    assert not is_complete_list_question(question)
    assert generate_answer_requirements(question) == []


def test_expands_generic_official_dismissal_question() -> None:
    queries = generate_search_queries("Khi nào sinh viên bị buộc thôi học?")

    assert queries[0] == "Khi nào sinh viên bị buộc thôi học?"
    assert "buộc thôi học do kết quả học tập yếu" in queries
    assert "cảnh báo kết quả học tập" in queries


def test_does_not_expand_outside_academic_context() -> None:
    question = "Điểm thấp nhất trên biểu đồ là bao nhiêu?"

    assert generate_search_queries(question) == [question]


def test_does_not_create_duplicate_queries() -> None:
    queries = generate_search_queries(
        "Sinh viên bị đuổi học hay buộc thôi học?"
    )

    folded = [query.casefold() for query in queries]
    assert len(folded) == len(set(folded))


def test_supports_extensible_graduation_and_failed_course_terms() -> None:
    graduation = generate_search_queries(
        "Sinh viên cần điều kiện gì để ra trường?"
    )
    failed_course = generate_search_queries(
        "Sinh viên thi trượt có phải học lại môn không?"
    )

    assert any("tốt nghiệp" in query for query in graduation[1:])
    assert any("học phần không đạt" in query for query in failed_course[1:])


def test_rejects_invalid_query_limit() -> None:
    try:
        generate_search_queries("câu hỏi", max_queries=0)
    except ValueError:
        return

    raise AssertionError("Expected ValueError to be raised")
