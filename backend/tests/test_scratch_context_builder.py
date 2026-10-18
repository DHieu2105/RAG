from app.rag.from_scratch.context_builder import build_context


def test_build_context() -> None:
    chunks = [
        {
            "article": 11,
            "clauses": [1, 2, 3],
            "page_start": 23,
            "text": "Sinh viên được cảnh báo sau mỗi học kỳ chính.",
        }
    ]

    context = build_context(chunks)

    assert context == (
        "[Tài liệu 1]\n"
        "Điều: 11\n"
        "Khoản: 1, 2, 3\n"
        "Trang PDF: 23\n"
        "Nội dung:\n"
        "Sinh viên được cảnh báo sau mỗi học kỳ chính."
    )


def test_build_context_joins_multiple_chunks() -> None:
    chunks = [
        {
            "article": 7,
            "clauses": [1],
            "page_start": 16,
            "text": "Nội dung thứ nhất.",
        },
        {
            "article": 8,
            "clauses": [2],
            "page_start": 18,
            "text": "Nội dung thứ hai.",
        },
    ]

    context = build_context(chunks)

    assert "[Tài liệu 1]" in context
    assert "[Tài liệu 2]" in context
    assert "\n\n[Tài liệu 2]" in context
    assert "Điều: 8" in context
    assert "Khoản: 2" in context
    assert "Trang PDF: 18" in context
    assert context.endswith("Nội dung thứ hai.")


def test_build_context_with_no_chunks() -> None:
    assert build_context([]) == ""


def test_build_context_supports_single_clause_field() -> None:
    chunks = [
        {
            "article": 11,
            "clause": 2,
            "page_start": 23,
            "text": "Sinh viên bị buộc thôi học trong các trường hợp sau.",
        }
    ]

    context = build_context(chunks)

    assert "Khoản: 2" in context


if __name__ == "__main__":
    test_build_context()
    test_build_context_joins_multiple_chunks()
    test_build_context_with_no_chunks()
    test_build_context_supports_single_clause_field()
    print("All tests passed!")
