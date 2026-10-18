"""Persist and restore the from-scratch TF-IDF index as JSON."""

import json
from pathlib import Path
from typing import Any

from app.rag.loader import BASE_DIR

from .embeddings import TfidfVectorizer


INDEX_VERSION = 1
INDEX_PATH = BASE_DIR / "data" / "processed" / "tfidf_index.json"


def save_index(
    chunks: list[dict],
    vectorizer: TfidfVectorizer,
    document_vectors: list[list[float]],
    path: Path = INDEX_PATH,
) -> None:
    """Save chunks and their fitted TF-IDF representation to ``path``."""
    if len(chunks) != len(document_vectors):
        raise ValueError("Số chunks và vectors không khớp")

    data = {
        "version": INDEX_VERSION,
        "chunks": chunks,
        "vocabulary": vectorizer.vocabulary,
        "idf": vectorizer.idf,
        "document_vectors": document_vectors,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def load_index(
    path: Path = INDEX_PATH,
) -> tuple[list[dict], TfidfVectorizer, list[list[float]]]:
    """Load a saved index and restore a fitted TF-IDF vectorizer."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    _validate_index_data(data)

    chunks = data["chunks"]
    document_vectors = data["document_vectors"]

    vectorizer = TfidfVectorizer()
    vectorizer.vocabulary = data["vocabulary"]
    vectorizer.idf = data["idf"]
    vectorizer._is_fitted = True

    return chunks, vectorizer, document_vectors


def _validate_index_data(data: Any) -> None:
    """Reject incompatible or internally inconsistent index files."""
    if not isinstance(data, dict):
        raise ValueError("Index phải là một JSON object")
    if data.get("version") != INDEX_VERSION:
        raise ValueError("Phiên bản index không được hỗ trợ")

    required_fields = (
        "chunks",
        "vocabulary",
        "idf",
        "document_vectors",
    )
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            "Index thiếu trường bắt buộc: " + ", ".join(missing_fields)
        )

    chunks = data["chunks"]
    vocabulary = data["vocabulary"]
    idf = data["idf"]
    document_vectors = data["document_vectors"]

    if not isinstance(chunks, list) or not all(
        isinstance(chunk, dict) for chunk in chunks
    ):
        raise ValueError("chunks phải là danh sách các object")
    if not isinstance(document_vectors, list) or not all(
        isinstance(vector, list) for vector in document_vectors
    ):
        raise ValueError("document_vectors phải là danh sách các vector")
    if len(chunks) != len(document_vectors):
        raise ValueError("Số chunks và vectors không khớp")

    if not isinstance(vocabulary, dict) or not all(
        isinstance(token, str)
        and isinstance(index, int)
        and not isinstance(index, bool)
        for token, index in vocabulary.items()
    ):
        raise ValueError("vocabulary không hợp lệ")

    vector_size = len(vocabulary)
    if sorted(vocabulary.values()) != list(range(vector_size)):
        raise ValueError("Các index trong vocabulary phải liên tục từ 0")

    if not isinstance(idf, dict) or set(idf) != set(vocabulary) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in idf.values()
    ):
        raise ValueError("idf không khớp với vocabulary")

    for vector in document_vectors:
        if len(vector) != vector_size:
            raise ValueError("Số chiều vector không khớp với vocabulary")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vector
        ):
            raise ValueError("Vector chỉ được chứa các giá trị số")
