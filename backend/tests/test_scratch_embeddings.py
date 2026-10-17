import math

from app.rag.from_scratch.embeddings import (
    TfidfVectorizer,
    build_vocabulary,
)
from app.rag.from_scratch.tokenizer import tokenize


def test_tokenize() -> None:
    assert tokenize("Sinh viên đăng ký tín chỉ") == [
        "sinh",
        "viên",
        "đăng",
        "ký",
        "tín",
        "chỉ",
    ]
    assert tokenize("  SINH\tviên\n") == ["sinh", "viên"]
    assert tokenize("") == []


def test_build_vocabulary() -> None:
    documents = [
        "sinh viên đăng ký tín chỉ",
        "sinh viên nhận học bổng",
        "quy định tín chỉ tối đa",
    ]

    assert build_vocabulary(documents) == {
        "sinh": 0,
        "viên": 1,
        "đăng": 2,
        "ký": 3,
        "tín": 4,
        "chỉ": 5,
        "nhận": 6,
        "học": 7,
        "bổng": 8,
        "quy": 9,
        "định": 10,
        "tối": 11,
        "đa": 12,
    }


def test_tfidf_vectorizer() -> None:
    documents = [
        "sinh viên sinh",
        "viên học",
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    sinh_idf = math.log(3 / 2) + 1
    assert math.isclose(vectorizer.idf["sinh"], sinh_idf)
    assert math.isclose(vectorizer.idf["viên"], 1.0)

    document_vectors = vectorizer.transform(documents)
    sinh_index = vectorizer.vocabulary["sinh"]
    vien_index = vectorizer.vocabulary["viên"]

    assert math.isclose(
        document_vectors[0][sinh_index],
        (2 / 3) * sinh_idf,
    )
    assert math.isclose(document_vectors[0][vien_index], 1 / 3)

    query_vector = vectorizer.transform(["sinh không-biết"])[0]
    assert len(query_vector) == len(vectorizer.vocabulary)
    assert math.isclose(query_vector[sinh_index], 0.5 * sinh_idf)


def test_empty_document() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["sinh viên"])

    assert vectorizer.transform([""]) == [[0.0, 0.0]]


def test_transform_before_fit() -> None:
    vectorizer = TfidfVectorizer()

    try:
        vectorizer.transform(["sinh viên"])
    except ValueError:
        return

    raise AssertionError("Expected ValueError to be raised")


if __name__ == "__main__":
    test_tokenize()
    test_build_vocabulary()
    test_tfidf_vectorizer()
    test_empty_document()
    test_transform_before_fit()
    print("All tests passed!")
