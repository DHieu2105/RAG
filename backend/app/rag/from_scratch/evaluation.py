"""Ground-truth dataset helpers and retrieval metrics."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable


VALID_LABEL_STATUSES = {"verified", "pending"}


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    relevant_chunk_ids: tuple[str, ...]
    answerable: bool | None
    category: str
    label_status: str
    notes: str = ""


@dataclass(frozen=True)
class RetrievalMetrics:
    evaluated_questions: int
    hit_at_1: float
    hit_at_5: float
    mrr: float


def stable_chunk_id(chunk: dict) -> str:
    """Create an ID that includes structure and exact chunk content."""
    identity = {
        "part": chunk.get("part"),
        "chapter": chunk.get("chapter"),
        "article": chunk.get("article"),
        "clauses": chunk.get("clauses", []),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "text": chunk.get("text", ""),
    }
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    part = str(chunk.get("part") or "none").replace(".", "-")
    article = str(chunk.get("article") or "none")
    page = str(chunk.get("page_start") or "none")
    return f"p{part}-a{article}-pg{page}-{digest}"


def load_evaluation_dataset(path: Path) -> list[EvaluationExample]:
    """Load and validate a JSONL evaluation dataset."""
    examples: list[EvaluationExample] = []
    seen_questions: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            data = json.loads(raw_line)
            required = {
                "question",
                "relevant_chunk_ids",
                "answerable",
                "category",
                "label_status",
            }
            missing = required - set(data)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing: {', '.join(sorted(missing))}"
                )

            question = data["question"]
            relevant_ids = data["relevant_chunk_ids"]
            answerable = data["answerable"]
            category = data["category"]
            label_status = data["label_status"]

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number} has an invalid question")
            if question in seen_questions:
                raise ValueError(f"Duplicate question on line {line_number}")
            seen_questions.add(question)
            if not isinstance(relevant_ids, list) or not all(
                isinstance(chunk_id, str) and chunk_id
                for chunk_id in relevant_ids
            ):
                raise ValueError(
                    f"Line {line_number} has invalid relevant_chunk_ids"
                )
            if answerable is not None and not isinstance(answerable, bool):
                raise ValueError(f"Line {line_number} has invalid answerable")
            if not isinstance(category, str) or not category:
                raise ValueError(f"Line {line_number} has invalid category")
            if label_status not in VALID_LABEL_STATUSES:
                raise ValueError(f"Line {line_number} has invalid label_status")
            if label_status == "verified" and answerable is None:
                raise ValueError(
                    f"Line {line_number}: verified labels need answerable"
                )
            if label_status == "verified" and answerable and not relevant_ids:
                raise ValueError(
                    f"Line {line_number}: answerable labels need relevant chunks"
                )
            if label_status == "verified" and answerable is False and relevant_ids:
                raise ValueError(
                    f"Line {line_number}: unanswerable labels cannot have chunks"
                )

            examples.append(
                EvaluationExample(
                    question=question,
                    relevant_chunk_ids=tuple(relevant_ids),
                    answerable=answerable,
                    category=category,
                    label_status=label_status,
                    notes=str(data.get("notes", "")),
                )
            )

    return examples


def first_relevant_rank(
    ranked_chunks: list[dict],
    relevant_chunk_ids: tuple[str, ...],
) -> int | None:
    """Return the first one-based rank matching verified ground truth."""
    relevant = set(relevant_chunk_ids)
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if stable_chunk_id(chunk) in relevant:
            return rank
    return None


def calculate_metrics(
    examples: list[EvaluationExample],
    retrieve: Callable[[str], list[dict]],
) -> tuple[RetrievalMetrics, dict[str, int | None]]:
    """Calculate metrics only for verified, answerable examples."""
    eligible = [
        example
        for example in examples
        if example.label_status == "verified" and example.answerable is True
    ]
    ranks: dict[str, int | None] = {}

    for example in eligible:
        ranks[example.question] = first_relevant_rank(
            retrieve(example.question),
            example.relevant_chunk_ids,
        )

    count = len(eligible)
    if count == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0), ranks

    hit_at_1 = sum(rank == 1 for rank in ranks.values()) / count
    hit_at_5 = sum(
        rank is not None and rank <= 5 for rank in ranks.values()
    ) / count
    mrr = sum(
        0.0 if rank is None else 1.0 / rank
        for rank in ranks.values()
    ) / count
    return RetrievalMetrics(count, hit_at_1, hit_at_5, mrr), ranks
