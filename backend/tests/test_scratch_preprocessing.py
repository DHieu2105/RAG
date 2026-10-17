from app.rag.from_scratch.preprocessing import (
    is_article_heading,
    is_clause_start,
    is_footer,
    merge_lines,
    normalize_line,
    preprocess_pages,
)


def test_normalize_line() -> None:
    assert normalize_line("   Sinh   viên   ") == "Sinh viên"
    assert normalize_line("Tối đa\t27 tín chỉ") == "Tối đa 27 tín chỉ"
    assert normalize_line("  ") == ""
    assert normalize_line("Điều 7. Đăng ký học tập") == "Điều 7. Đăng ký học tập"


def test_is_footer() -> None:
    assert is_footer("14 * STSV") is True
    assert is_footer("STSV * 15") is True
    assert is_footer("  16   *   STSV ") is True
    assert is_footer("Điều 7. Đăng ký học tập") is False
    assert is_footer("27 tín chỉ") is False
    assert is_footer("") is False


def test_is_article_heading() -> None:
    assert is_article_heading("Điều 7. Tổ chức đăng ký học tập")
    assert is_article_heading("Điều 15. Công nhận tốt nghiệp")
    assert is_article_heading("Điều 8")
    assert not is_article_heading("5. Khối lượng học tập")
    assert not is_article_heading("Sinh viên đăng ký tín chỉ")
    assert not is_article_heading("")


def test_is_clause_start() -> None:
    assert is_clause_start("1. Sinh viên đăng ký")
    assert is_clause_start("15. Công nhận tốt nghiệp")
    assert not is_clause_start("Điều 7. Đăng ký học tập")
    assert not is_clause_start("27 tín chỉ")
    assert not is_clause_start("")


def test_merge_lines() -> None:
    lines = [
        "Sinh viên phải đăng ký",
        "tối thiểu 12 tín chỉ",
        "và tối đa 27 tín chỉ.",
    ]
    result = merge_lines(lines)

    assert result == (
        "Sinh viên phải đăng ký "
        "tối thiểu 12 tín chỉ "
        "và tối đa 27 tín chỉ."
    )
    assert merge_lines([]) == ""
    assert merge_lines(["14 * STSV", "Nội dung"]) == "Nội dung"


def test_preprocess_pages() -> None:
    pages = [
        {
            "page_number": 16,
            "text": (
                "Điều 7. Đăng ký học tập\n"
                "1. Sinh viên đăng ký\n"
                "tối đa 27 tín chỉ.\n"
                "2. Sinh viên có thể\n"
                "rút bớt học phần.\n"
                "14 * STSV"
            ),
        }
    ]

    blocks = preprocess_pages(pages)

    assert len(blocks) == 2
    assert blocks[0]["article"] == 7
    assert blocks[0]["article_title"] == "Điều 7. Đăng ký học tập"
    assert blocks[0]["clause"] == 1
    assert blocks[0]["page_start"] == 16
    assert blocks[0]["page_end"] == 16
    assert blocks[0]["text"] == (
        "1. Sinh viên đăng ký tối đa 27 tín chỉ."
    )
    assert blocks[1]["clause"] == 2


def test_preprocess_pages_keeps_state_across_pages() -> None:
    pages = [
        {
            "page_number": 16,
            "text": (
                "Điều 7. Đăng ký học tập\n"
                "1. Sinh viên đăng ký\n"
                "14 * STSV"
            ),
        },
        {
            "page_number": 17,
            "text": (
                "tối đa 27 tín chỉ.\n"
                "2. Sinh viên có thể rút học phần."
            ),
        },
    ]

    blocks = preprocess_pages(pages)

    assert blocks[0]["article"] == 7
    assert blocks[0]["clause"] == 1
    assert blocks[0]["page_start"] == 16
    assert blocks[0]["page_end"] == 17
    assert blocks[0]["text"] == (
        "1. Sinh viên đăng ký tối đa 27 tín chỉ."
    )
    assert blocks[1]["page_start"] == 17


if __name__ == "__main__":
    test_normalize_line()
    test_is_footer()
    test_is_article_heading()
    test_is_clause_start()
    test_merge_lines()
    test_preprocess_pages()
    test_preprocess_pages_keeps_state_across_pages()
    print("All tests passed!")
