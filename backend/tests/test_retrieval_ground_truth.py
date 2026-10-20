import json
from pathlib import Path

from app.rag.from_scratch.evaluation import (
    load_evaluation_dataset,
    stable_chunk_id,
)
from app.rag.from_scratch.index_store import load_index


DATASET_PATH = Path(__file__).parent / "data" / "retrieval_evaluation.jsonl"


def test_dataset_has_existing_questions_and_new_colloquial_set() -> None:
    raw_records = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = load_evaluation_dataset(DATASET_PATH)

    assert len(examples) == len(raw_records)
    assert sum(record.get("origin") == "existing_20" for record in raw_records) == 20
    assert sum(
        record.get("origin") == "new_colloquial" for record in raw_records
    ) >= 10
    assert all(
        not example.relevant_chunk_ids
        for example in examples
        if example.answerable is False
    )
    assert all(
        example.label_status == "pending"
        for example in examples
        if example.answerable is None
    )


def test_all_verified_relevant_chunk_ids_exist_in_current_index() -> None:
    examples = load_evaluation_dataset(DATASET_PATH)
    chunks, _, _ = load_index()
    known_ids = {stable_chunk_id(chunk) for chunk in chunks}

    assert all(
        chunk_id in known_ids
        for example in examples
        if example.label_status == "verified"
        for chunk_id in example.relevant_chunk_ids
    )
