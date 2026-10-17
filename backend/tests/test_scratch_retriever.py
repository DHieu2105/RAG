import math

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.retriever import (
    cosine_similarity,
    dot_product,
    top_k_search,
    vector_norm,
)


def assert_raises_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return

    raise AssertionError("Expected ValueError to be raised")


def test_dot_product() -> None:
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32
    assert dot_product([1, 0], [0, 1]) == 0
    assert dot_product([2, 3], [2, 3]) == 13
    assert dot_product([], []) == 0


def test_dot_product_rejects_different_dimensions() -> None:
    assert_raises_value_error(
        lambda: dot_product([1, 2], [1, 2, 3])
    )


def test_vector_norm() -> None:
    assert vector_norm([3, 4]) == 5
    assert vector_norm([0, 0]) == 0
    assert vector_norm([1, 0, 0]) == 1
    assert vector_norm([]) == 0


def test_cosine_similarity() -> None:
    assert math.isclose(
        cosine_similarity([1, 0], [1, 0]),
        1.0,
    )
    assert math.isclose(
        cosine_similarity([1, 0], [0, 1]),
        0.0,
    )
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_top_k_search() -> None:
    query_vector = [1.0, 0.0]
    document_vectors = [
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    chunks = [
        {"text": "partly relevant", "article": 1},
        {"text": "most relevant", "article": 2},
        {"text": "not relevant", "article": 3},
    ]

    results = top_k_search(
        query_vector,
        document_vectors,
        chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["article"] == 2
    assert math.isclose(results[0]["score"], 1.0)
    assert results[1]["article"] == 1
    assert all(result["score"] > 0 for result in results)
    assert "score" not in chunks[0]


def test_top_k_search_validates_arguments() -> None:
    assert_raises_value_error(
        lambda: top_k_search([1.0], [[1.0]], [], top_k=1)
    )
    assert_raises_value_error(
        lambda: top_k_search([1.0], [[1.0]], [{}], top_k=0)
    )
    assert_raises_value_error(
        lambda: top_k_search([1.0], [[1.0, 0.0]], [{}], top_k=1)
    )


def test_retriever_with_tfidf() -> None:
    chunks = [
        {"text": "sinh viên đăng ký tín chỉ", "article": 7},
        {"text": "sinh viên được xét học bổng", "article": 12},
        {"text": "quy định tín chỉ tối đa", "article": 7},
    ]
    documents = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    document_vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform(["tín chỉ tối đa"])[0]

    results = top_k_search(
        query_vector,
        document_vectors,
        chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == "quy định tín chỉ tối đa"
    assert results[0]["score"] > results[1]["score"]
    assert results[1]["text"] == "sinh viên đăng ký tín chỉ"


if __name__ == "__main__":
    test_dot_product()
    test_dot_product_rejects_different_dimensions()
    test_vector_norm()
    test_cosine_similarity()
    test_top_k_search()
    test_top_k_search_validates_arguments()
    test_retriever_with_tfidf()
    print("All tests passed!")
