"""Vector similarity and retrieval algorithms implemented from scratch."""

import math


DEFAULT_RRF_K = 60


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


def chunk_identity(chunk: dict) -> tuple:
    """Return a stable identity without conflating equal article numbers."""
    clauses = chunk.get("clauses")
    if clauses is None:
        clause = chunk.get("clause")
        clauses = [clause] if clause is not None else []

    return (
        chunk.get("part"),
        chunk.get("part_title"),
        chunk.get("chapter"),
        chunk.get("chapter_title"),
        chunk.get("article"),
        chunk.get("article_title"),
        tuple(clauses),
        chunk.get("page_start"),
        chunk.get("page_end"),
        chunk.get("text"),
    )


def reciprocal_rank_fusion(
    ranked_results: list[list[dict]],
    *,
    weights: list[float] | None = None,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
    source_score_name: str = "source_score",
) -> list[dict]:
    """Fuse ranked lists using weighted Reciprocal Rank Fusion."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    if weights is None:
        weights = [1.0] * len(ranked_results)
    if len(weights) != len(ranked_results):
        raise ValueError("weights and ranked_results must have the same length")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("weights must be greater than 0")
    if not source_score_name:
        raise ValueError("source_score_name must not be empty")

    fused: dict[tuple, dict] = {}
    first_seen = 0

    for query_index, results in enumerate(ranked_results):
        weight = weights[query_index]
        seen_in_query: set[tuple] = set()

        for rank, result in enumerate(results, start=1):
            identity = chunk_identity(result)
            if identity in seen_in_query:
                continue
            seen_in_query.add(identity)

            entry = fused.get(identity)
            if entry is None:
                entry = {
                    "chunk": {
                        key: value
                        for key, value in result.items()
                        if key != "score"
                    },
                    "score": 0.0,
                    "best_source_score": result.get("score", 0.0),
                    "best_rank": rank,
                    "first_seen": first_seen,
                    "matched_queries": 0,
                }
                fused[identity] = entry
                first_seen += 1

            entry["score"] += weight / (rrf_k + rank)
            entry["best_source_score"] = max(
                entry["best_source_score"],
                result.get("score", 0.0),
            )
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["matched_queries"] += 1

    ordered = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["score"],
            entry["best_rank"],
            entry["first_seen"],
        ),
    )

    results: list[dict] = []
    for entry in ordered[:top_k]:
        results.append(
            {
                **entry["chunk"],
                "score": entry["score"],
                source_score_name: entry["best_source_score"],
                "matched_queries": entry["matched_queries"],
            }
        )
    return results

