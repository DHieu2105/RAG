import math

from app.rag.from_scratch.bm25 import BM25Index, build_bm25_index


def test_bm25_prefers_repeated_matching_term() -> None:
    index = BM25Index()
    index.fit(["cảnh báo học tập học tập", "quy định học tập"])

    scores = index.scores("học tập")

    assert scores[0] > scores[1] > 0.0


def test_bm25_applies_document_length_normalization() -> None:
    index = BM25Index()
    index.fit(
        [
            "buộc thôi học",
            "buộc thôi học nội dung bổ sung không liên quan rất dài",
        ]
    )

    scores = index.scores("buộc thôi học")

    assert scores[0] > scores[1]


def test_bm25_search_returns_ranked_chunk_copies() -> None:
    chunks = [
        {"text": "điều kiện tốt nghiệp", "article": 15},
        {"text": "đăng ký tín chỉ", "article": 7},
    ]
    index = build_bm25_index(chunks)

    results = index.search("tốt nghiệp", chunks)

    assert len(results) == 1
    assert results[0]["article"] == 15
    assert results[0]["score"] > 0.0
    assert "score" not in chunks[0]


def test_bm25_indexes_structural_titles() -> None:
    chunks = [
        {
            "article_title": "Điều 3. Các đối tượng được miễn học phí",
            "text": "1. Người có công với cách mạng.",
        },
        {
            "article_title": "Điều 4. Trợ cấp xã hội",
            "text": "2. Sinh viên có hoàn cảnh khó khăn.",
        },
    ]
    index = build_bm25_index(chunks)

    results = index.search("đối tượng miễn học phí", chunks)

    assert results[0]["article_title"].startswith("Điều 3")


def test_bm25_uses_standard_smoothed_idf() -> None:
    index = BM25Index()
    index.fit(["a b", "a c"])

    expected = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    assert math.isclose(index.idf["b"], expected)


def test_bm25_validates_state_and_arguments() -> None:
    try:
        BM25Index().scores("query")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected scores before fit to fail")

    for factory in (
        lambda: BM25Index(k1=0),
        lambda: BM25Index(b=-0.1),
        lambda: BM25Index(b=1.1),
        lambda: BM25Index().fit([]),
    ):
        try:
            factory()
        except ValueError:
            continue
        raise AssertionError("Expected invalid BM25 configuration to fail")
