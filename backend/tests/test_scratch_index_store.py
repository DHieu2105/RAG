import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.index_store import load_index, save_index


def assert_raises_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error)
        return

    raise AssertionError("Expected ValueError to be raised")


def test_save_and_load_index() -> None:
    chunks = [
        {"text": "sinh viên đăng ký tín chỉ", "article": 1},
        {"text": "sinh viên nhận học bổng", "article": 2},
    ]
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    vectors = vectorizer.transform(texts)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "tfidf_index.json"

        save_index(chunks, vectorizer, vectors, path)
        loaded_chunks, loaded_vectorizer, loaded_vectors = load_index(path)

        assert loaded_chunks == chunks
        assert loaded_vectors == vectors
        assert loaded_vectorizer.vocabulary == vectorizer.vocabulary
        assert loaded_vectorizer.idf == vectorizer.idf
        assert loaded_vectorizer.transform(["đăng ký tín chỉ"]) == (
            vectorizer.transform(["đăng ký tín chỉ"])
        )

        saved_data = json.loads(path.read_text(encoding="utf-8"))
        assert saved_data["version"] == 1
        assert saved_data["chunks"][0]["text"] == chunks[0]["text"]


def test_save_index_rejects_mismatched_lengths() -> None:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["một tài liệu"])

    with TemporaryDirectory() as directory:
        assert_raises_value_error(
            lambda: save_index(
                [{"text": "một tài liệu"}],
                vectorizer,
                [],
                Path(directory) / "index.json",
            ),
            "Số chunks và vectors không khớp",
        )


def test_load_index_rejects_wrong_vector_size() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "index.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "chunks": [{"text": "một tài liệu"}],
                    "vocabulary": {"một": 0, "tài": 1},
                    "idf": {"một": 1.0, "tài": 1.0},
                    "document_vectors": [[1.0]],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        assert_raises_value_error(
            lambda: load_index(path),
            "Số chiều vector",
        )


if __name__ == "__main__":
    test_save_and_load_index()
    test_save_index_rejects_mismatched_lengths()
    test_load_index_rejects_wrong_vector_size()
    print("All tests passed!")
