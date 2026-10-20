from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.pipeline import ask_hybrid


def _build_components(chunks: list[dict]):
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunk["text"] for chunk in chunks])
    vectors = vectorizer.transform([chunk["text"] for chunk in chunks])
    bm25_index = build_bm25_index(chunks)
    return vectorizer, vectors, bm25_index


def test_hybrid_keeps_same_article_in_different_parts_distinct() -> None:
    chunks = [
        {
            "part": "2.1",
            "article": 3,
            "clauses": [1],
            "page_start": 12,
            "text": "quy định chương trình đào tạo",
        },
        {
            "part": "2.6",
            "article": 3,
            "clauses": [1],
            "page_start": 47,
            "text": "quy định miễn giảm học phí",
        },
    ]
    vectorizer, vectors, bm25_index = _build_components(chunks)

    results = ask_hybrid(
        "quy định",
        chunks,
        vectorizer,
        vectors,
        bm25_index,
    )

    assert len(results) == 2
    assert {result["part"] for result in results} == {"2.1", "2.6"}


def test_hybrid_reranks_only_when_query_is_expanded() -> None:
    chunks = [
        {
            "part": "2.1",
            "article": 11,
            "clauses": [1, 2],
            "page_start": 23,
            "text": (
                "buộc thôi học trong các trường hợp sau "
                "do kết quả học tập yếu"
            ),
        },
        {
            "part": "2.1",
            "article": 9,
            "clauses": [3],
            "page_start": 19,
            "text": "điểm thấp và điểm F của học phần",
        },
    ]
    vectorizer, vectors, bm25_index = _build_components(chunks)

    plain = ask_hybrid(
        "điểm F của học phần",
        chunks,
        vectorizer,
        vectors,
        bm25_index,
    )
    expanded = ask_hybrid(
        "Điểm thấp có bị đuổi học không?",
        chunks,
        vectorizer,
        vectors,
        bm25_index,
    )

    assert "retrieval_score" not in plain[0]
    assert "retrieval_score" in expanded[0]
    assert expanded[0]["article"] == 11


def test_hybrid_results_are_stable() -> None:
    chunks = [
        {"text": "đăng ký tối đa tín chỉ", "article": 7},
        {"text": "điều kiện tốt nghiệp", "article": 15},
    ]
    vectorizer, vectors, bm25_index = _build_components(chunks)

    first = ask_hybrid(
        "đăng ký tối đa tín chỉ",
        chunks,
        vectorizer,
        vectors,
        bm25_index,
    )
    second = ask_hybrid(
        "đăng ký tối đa tín chỉ",
        chunks,
        vectorizer,
        vectors,
        bm25_index,
    )

    assert first == second
