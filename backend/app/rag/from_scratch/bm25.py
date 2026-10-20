"""A small BM25 retriever implemented with the Python standard library."""

from collections import Counter
import math

from .tokenizer import tokenize


class BM25Index:
    """Store corpus statistics and rank documents with Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0.0:
            raise ValueError("k1 must be greater than 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self.document_term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        self.average_document_length = 0.0
        self.idf: dict[str, float] = {}
        self._is_fitted = False

    def fit(self, documents: list[str]) -> None:
        """Learn document frequencies and length statistics."""
        if not documents:
            raise ValueError("documents must not be empty")

        self.document_term_frequencies = []
        self.document_lengths = []
        document_frequency: Counter[str] = Counter()

        for document in documents:
            tokens = tokenize(document)
            frequencies = Counter(tokens)
            self.document_term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())

        document_count = len(documents)
        self.average_document_length = (
            sum(self.document_lengths) / document_count
        )
        self.idf = {
            token: math.log(
                1.0
                + (
                    document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }
        self._is_fitted = True

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score for every fitted document."""
        if not self._is_fitted:
            raise ValueError("fit must be called before scores")

        query_terms = set(tokenize(query))
        scores: list[float] = []

        for frequencies, document_length in zip(
            self.document_term_frequencies,
            self.document_lengths,
        ):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                length_ratio = (
                    document_length / self.average_document_length
                    if self.average_document_length > 0.0
                    else 0.0
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += self.idf[term] * (
                    term_frequency * (self.k1 + 1.0)
                ) / denominator
            scores.append(score)

        return scores

    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Return the highest-scoring chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if len(chunks) != len(self.document_term_frequencies):
            raise ValueError("chunks and fitted documents must have equal length")

        results = [
            {**chunk, "score": score}
            for chunk, score in zip(chunks, self.scores(query))
            if score > 0.0
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


def build_bm25_index(chunks: list[dict]) -> BM25Index:
    """Build a BM25 index from structural titles and chunk text."""
    index = BM25Index()
    documents = []
    for chunk in chunks:
        fields = (
            chunk.get("part_title"),
            chunk.get("chapter_title"),
            chunk.get("article_title"),
            chunk["text"],
        )
        documents.append(
            " ".join(str(field) for field in fields if field)
        )
    index.fit(documents)
    return index
