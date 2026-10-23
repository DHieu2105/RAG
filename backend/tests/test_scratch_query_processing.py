import unicodedata

from app.rag.from_scratch.query_processing import (
    generate_answer_requirements,
    generate_search_queries,
    is_complete_list_question,
    normalize_query,
)
from app.rag.from_scratch.query_understanding import QueryIntent, analyze_query


def test_normalize_query_normalizes_unicode_and_whitespace() -> None:
    decomposed = unicodedata.normalize(
        "NFD",
        "  Điểm   thấp\n có bị đuổi học không?  ",
    )

    normalized = normalize_query(decomposed)

    assert normalized == "Điểm thấp có bị đuổi học không?"
    assert unicodedata.is_normalized("NFC", normalized)


def test_detects_scenario_and_condition_intents() -> None:
    analysis = analyze_query("Em có đủ điều kiện tốt nghiệp không?")

    assert analysis.has_intent(QueryIntent.FACTUAL_LOOKUP)
    assert analysis.has_intent(QueryIntent.CONDITIONS)
    assert analysis.has_intent(QueryIntent.SCENARIO_ELIGIBILITY)


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
    assert not any("cảnh báo kết quả học tập" == query for query in queries)


def test_expands_procedure_without_adding_neighboring_rules() -> None:
    queries = generate_search_queries("Quy trình đăng ký tín chỉ như thế nào?")

    assert queries[0] == "Quy trình đăng ký tín chỉ như thế nào?"
    assert any(query.startswith("hướng dẫn") for query in queries)
    assert not any("khối lượng học tập" in query for query in queries)


def test_procedure_requirements_are_topic_independent() -> None:
    requirements = generate_answer_requirements("Quy trình bất kỳ như thế nào?")

    assert any("đúng thứ tự" in item for item in requirements)
    assert not any("tín chỉ" in item or "học phí" in item for item in requirements)


def test_scenario_requirements_do_not_embed_an_answer_chain() -> None:
    requirements = generate_answer_requirements(
        "Điểm thấp quá có bị đuổi học không?"
    )

    assert any("điều kiện" in item for item in requirements)
    assert not any("số lần cảnh báo" in item for item in requirements)


def test_enumeration_does_not_expand_unasked_benefit_levels() -> None:
    question = "Đối tượng nào được miễn, giảm học phí?"

    queries = generate_search_queries(question)

    assert queries[0] == question
    assert not any("100%" in query or "70%" in query or "50%" in query for query in queries)
    assert is_complete_list_question(question)


def test_enumeration_requirements_use_context_coverage() -> None:
    requirements = generate_answer_requirements(
        "Những đối tượng sinh viên nào được miễn học phí?"
    )

    assert any("checklist coverage" in item for item in requirements)
    assert not any("sáu" in item or "70%" in item or "50%" in item for item in requirements)


def test_personal_tuition_question_does_not_require_full_policy_list() -> None:
    question = "Em có được miễn học phí không?"

    assert not is_complete_list_question(question)
    requirements = generate_answer_requirements(question)
    assert any("thiếu dữ kiện" in item for item in requirements)


def test_condition_expansion_preserves_the_asked_scope() -> None:
    queries = generate_search_queries("Khi nào sinh viên bị buộc thôi học?")

    assert queries[0] == "Khi nào sinh viên bị buộc thôi học?"
    assert any("điều kiện" in query and "buộc thôi học" in query for query in queries)
    assert not any("cảnh báo" in query for query in queries)


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


def test_short_tuition_list_questions() -> None:
    questions = [
        "ai được miễn giảm học phí",
        "những ai được miễn giảm học phí",
        "đối tượng nào được miễn, giảm học phí",
        "liệt kê các đối tượng được miễn giảm học phí",
    ]

    for question in questions:
        assert is_complete_list_question(question)
        requirements = generate_answer_requirements(question)
        assert any("checklist coverage" in item for item in requirements)
        assert not any("sáu" in item or "70%" in item or "50%" in item for item in requirements)


def test_specific_full_tuition_exemption_stays_scoped_to_100_percent() -> None:
    question = "Những ai được miễn 100% học phí?"

    assert is_complete_list_question(question)
    queries = generate_search_queries(question)
    requirements = generate_answer_requirements(question)

    assert any("miễn 100%" in query for query in queries)
    assert not any("70%" in query or "50%" in query for query in queries)
    assert any("checklist coverage" in item for item in requirements)
    assert not any("70%" in item or "50%" in item for item in requirements)


def test_social_allowance_uses_the_same_generic_enumeration_requirement() -> None:
    question = "Ai được cấp trợ cấp xã hội?"

    assert is_complete_list_question(question)
    requirements = generate_answer_requirements(question)
    assert any("checklist coverage" in item for item in requirements)


def test_recognizes_all_supported_query_shapes() -> None:
    examples = {
        QueryIntent.FACTUAL_LOOKUP: "Hạng tốt nghiệp là gì?",
        QueryIntent.ENUMERATION: "Liệt kê tất cả trường hợp được nghỉ học.",
        QueryIntent.CONDITIONS: "Điều kiện tốt nghiệp là gì?",
        QueryIntent.COMPARISON: "Phân biệt miễn học phí và giảm học phí.",
        QueryIntent.PROCEDURE: "Thủ tục đăng ký học như thế nào?",
        QueryIntent.NUMERIC_THRESHOLD: "Tối thiểu bao nhiêu tín chỉ?",
        QueryIntent.SCENARIO_ELIGIBILITY: "Em có đủ điều kiện tốt nghiệp không?",
    }

    for intent, question in examples.items():
        assert analyze_query(question).has_intent(intent)
