from app.rag.from_scratch.chunker import structure_chunk
from app.rag.from_scratch.context_builder import build_context
from app.rag.from_scratch.preprocessing import (
    is_part_heading,
    preprocess_pages,
)


def _sample_pages() -> list[dict]:
    return [
        {
            "page_number": 41,
            "text": (
                "PHẦN 2.4. QUY ĐỊNH VỀ KỶ LUẬT SINH VIÊN\n"
                "Điều 25. Thời hiệu quyết định kỷ luật\n"
                "4. Nội dung cuối của phần kỷ luật.\n"
                "PHẦN 2.5. QUY ĐỊNH VỀ HỌC BỔNG\n"
                "1. Học bổng khuyến khích học tập\n"
                "Sinh viên đủ điều kiện được xét học bổng."
            ),
        },
        {
            "page_number": 46,
            "text": (
                "2. Học bổng chính sách\n"
                "Sinh viên hệ cử tuyển được cấp học bổng chính sách."
            ),
        },
    ]


def test_part_heading_requires_a_number() -> None:
    assert is_part_heading("PHẦN 2.5. QUY ĐỊNH VỀ HỌC BỔNG")
    assert is_part_heading("Phần 3: Hướng dẫn thực hiện")
    assert not is_part_heading("phần ngoại ngữ chuyên ngành")


def test_preprocessing_resets_article_and_clause_at_part_boundary() -> None:
    blocks = preprocess_pages(_sample_pages())

    discipline = next(block for block in blocks if block.get("part") == "2.4")
    scholarship = [block for block in blocks if block.get("part") == "2.5"]

    assert discipline["article"] == 25
    assert discipline["clause"] == 4
    assert scholarship[0]["article"] is None
    assert scholarship[0]["clause"] == 1
    assert scholarship[1]["clause"] == 2
    assert scholarship[1]["page_start"] == 46
    assert all("kỷ luật" not in block["text"] for block in scholarship)


def test_chunker_never_combines_different_parts_or_articles() -> None:
    blocks = preprocess_pages(_sample_pages())
    chunks = structure_chunk(blocks, max_size=1000)

    assert {chunk.get("part") for chunk in chunks} == {"2.4", "2.5"}
    assert all(
        not (
            "Nội dung cuối của phần kỷ luật" in chunk["text"]
            and "Học bổng khuyến khích học tập" in chunk["text"]
        )
        for chunk in chunks
    )


def test_chunker_separates_repeated_article_numbers_by_part() -> None:
    blocks = [
        {
            "part": "2.1",
            "article": 3,
            "clause": 1,
            "page_start": 12,
            "page_end": 12,
            "text": "Điều 3 thuộc quy chế đào tạo.",
        },
        {
            "part": "2.6",
            "article": 3,
            "clause": 1,
            "page_start": 47,
            "page_end": 47,
            "text": "Điều 3 thuộc quy định học phí.",
        },
    ]

    chunks = structure_chunk(blocks, max_size=1000)

    assert len(chunks) == 2
    assert [chunk["part"] for chunk in chunks] == ["2.1", "2.6"]


def test_context_contains_exact_available_structure_and_page_range() -> None:
    context = build_context(
        [
            {
                "part": "2.1",
                "part_title": "PHẦN 2.1. QUY CHẾ ĐÀO TẠO ĐẠI HỌC",
                "chapter": "III",
                "chapter_title": (
                    "Chương III ĐÁNH GIÁ KẾT QUẢ HỌC TẬP "
                    "VÀ CẤP BẰNG TỐT NGHIỆP"
                ),
                "article": 11,
                "article_title": "Điều 11. Xử lý kết quả học tập",
                "clauses": [1, 2],
                "page_start": 23,
                "page_end": 24,
                "text": "Nội dung Điều 11.",
            }
        ]
    )

    assert "Phần: 2.1 — PHẦN 2.1. QUY CHẾ ĐÀO TẠO ĐẠI HỌC" in context
    assert "Chương: III — Chương III ĐÁNH GIÁ KẾT QUẢ" in context
    assert "Điều: 11 — Điều 11. Xử lý kết quả học tập" in context
    assert "Khoản: 1, 2" in context
    assert "Trang PDF: 23-24" in context


if __name__ == "__main__":
    test_part_heading_requires_a_number()
    test_preprocessing_resets_article_and_clause_at_part_boundary()
    test_chunker_never_combines_different_parts_or_articles()
    test_chunker_separates_repeated_article_numbers_by_part()
    test_context_contains_exact_available_structure_and_page_range()
    print("All tests passed!")
