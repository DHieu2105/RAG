"""A small TF-IDF vectorizer implemented with the Python standard library."""

import math

from .tokenizer import tokenize


def build_vocabulary(documents: list[str]) -> dict[str, int]:
    """Map each distinct token to an index in first-seen order."""
    vocabulary: dict[str, int] = {}

    for document in documents:
        for token in tokenize(document):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


class TfidfVectorizer:
    """Learn TF-IDF weights and transform text into dense vectors."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._is_fitted = False

    def fit(self, documents: list[str]) -> None:
        """Learn the vocabulary and smoothed IDF values from documents."""
        self.vocabulary = build_vocabulary(documents)
        document_frequency = {
            token: 0 for token in self.vocabulary
        }

        for document in documents:
            unique_tokens = set(tokenize(document))
            for token in unique_tokens:
                document_frequency[token] += 1

        total_documents = len(documents)
        self.idf = {
            token: math.log(
                (total_documents + 1) / (frequency + 1)
            ) + 1
            for token, frequency in document_frequency.items()
        }
        self._is_fitted = True

    def transform(self, documents: list[str]) -> list[list[float]]:
        """Transform documents using the vocabulary and IDF learned by fit."""
        if not self._is_fitted:
            raise ValueError("fit must be called before transform")

        vectors: list[list[float]] = []

        for document in documents:
            tokens = tokenize(document)
            vector = [0.0] * len(self.vocabulary)

            if tokens:
                token_counts: dict[str, int] = {}
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1

                total_tokens = len(tokens)
                for token, count in token_counts.items():
                    if token in self.vocabulary:
                        index = self.vocabulary[token]
                        term_frequency = count / total_tokens
                        vector[index] = term_frequency * self.idf[token]

            vectors.append(vector)

        return vectors

