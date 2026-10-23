from app.rag.from_scratch.grounding import (
    AnswerCoverageError,
    CitationValidationError,
    context_is_relevant,
    extract_coverage_units,
    prepare_structured_context,
    validate_answer_coverage,
    validate_claim_faithfulness,
)
from app.rag.from_scratch.query_understanding import analyze_query


def test_structure_expansion_keeps_only_asked_clause_and_continuations() -> None:
    chunks = [
        {
            "part": "2.6",
            "article": 3,
            "article_title": "Điều 3. Miễn, giảm học phí",
            "clauses": [1],
            "page_start": 47,
            "text": "1. Đối tượng được miễn học phí: a. Nhóm Alpha. b. Nhóm Beta.",
        },
        {
            "part": "2.6",
            "article": 3,
            "article_title": "Điều 3. Miễn, giảm học phí",
            "clauses": [2],
            "page_start": 48,
            "text": "2. Đối tượng được giảm học phí: a. Nhóm Gamma. b. Nhóm Delta.",
        },
    ]
    selected = prepare_structured_context(
        analyze_query("Những đối tượng nào được miễn học phí?"),
        chunks,
        chunks,
        max_chars=10_000,
        max_chunks=10,
    )

    assert [chunk["clauses"] for chunk in selected] == [[1]]


def test_coverage_units_come_from_actual_list_items() -> None:
    results = [
        {
            "article": 8,
            "clauses": [1],
            "text": "1. Danh sách: a. Mục Alpha. b. Mục Beta. c. Mục Gamma.",
        }
    ]
    analysis = analyze_query("Liệt kê các mục trong danh sách.")
    units = extract_coverage_units(analysis, results)

    assert [unit.key for unit in units] == [
        "S1-item-a",
        "S1-item-b",
        "S1-item-c",
    ]


def test_coverage_requires_evidence_from_the_unit_document() -> None:
    results = [
        {"article": 8, "clauses": [1], "text": "a. Alpha. b. Beta."},
        {"article": 8, "clauses": [1], "text": "c. Gamma. d. Delta."},
    ]
    units = extract_coverage_units(analyze_query("Liệt kê các mục."), results)
    citations = [
        {"document": 1, "claim": "Alpha", "evidence": "a. Alpha."},
        {"document": 1, "claim": "Beta", "evidence": "b. Beta."},
        {"document": 1, "claim": "Gamma", "evidence": "c. Gamma."},
        {"document": 1, "claim": "Delta", "evidence": "d. Delta."},
    ]

    try:
        validate_answer_coverage(units, citations)
    except AnswerCoverageError as error:
        assert "item-c" in str(error) and "item-d" in str(error)
        return
    raise AssertionError("Expected AnswerCoverageError")


def test_relevance_rejects_out_of_vocabulary_subject() -> None:
    analysis = analyze_query("Trường có cấp laptop miễn phí không?")
    results = [{"text": "Nhà trường hướng dẫn thanh toán học phí."}]

    assert not context_is_relevant(
        analysis,
        results,
        known_terms={"trường", "cấp", "miễn", "phí"},
    )


def test_faithfulness_rejects_negative_claim_without_negative_evidence() -> None:
    analysis = analyze_query("Trường có cấp laptop miễn phí không?")
    results = [
        {
            "part": "3",
            "article": None,
            "clauses": [],
            "page_start": 75,
            "text": "Sinh viên có thể thanh toán học phí bằng mã QR.",
        }
    ]
    citations = [
        {
            "claim": "Trường không cấp laptop miễn phí.",
            "document": 1,
            "clauses": [],
            "evidence": "Sinh viên có thể thanh toán học phí bằng mã QR.",
        }
    ]

    try:
        validate_claim_faithfulness(analysis, results, citations)
    except CitationValidationError:
        return
    raise AssertionError("Expected CitationValidationError")
