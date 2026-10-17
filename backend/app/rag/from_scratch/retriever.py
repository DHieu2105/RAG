"""Vector similarity and retrieval algorithms implemented from scratch."""

import math


def dot_product(a: list[float], b: list[float]) -> float:
    """Return the dot product of two vectors with equal dimensions."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same number of dimensions")

    total = 0.0
    for x, y in zip(a, b):
        total += x * y

    return total


def vector_norm(vector: list[float]) -> float:
    """Return the Euclidean norm of a vector."""
    sum_of_squares = 0.0
    for value in vector:
        sum_of_squares += value * value

    return math.sqrt(sum_of_squares)


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """Return the cosine similarity between two vectors."""
    product = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return product / (norm_a * norm_b)


def top_k_search(
    query_vector: list[float],
    document_vectors: list[list[float]],
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Return the highest-scoring chunks with positive cosine similarity."""
    if len(document_vectors) != len(chunks):
        raise ValueError("document_vectors and chunks must have the same length")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    results: list[dict] = []

    for index, document_vector in enumerate(document_vectors):
        score = cosine_similarity(query_vector, document_vector)
        if score <= 0.0:
            continue

        result = {
            **chunks[index],
            "score": score,
        }
        results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )
    return results[:top_k]

