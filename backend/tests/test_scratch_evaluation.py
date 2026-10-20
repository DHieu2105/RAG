import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.from_scratch.evaluation import (
    EvaluationExample,
    calculate_metrics,
    load_evaluation_dataset,
    stable_chunk_id,
)


def test_stable_chunk_id_distinguishes_parts_and_content() -> None:
    base = {
        "part": "2.1",
        "article": 3,
        "clauses": [1],
        "page_start": 12,
        "page_end": 12,
        "text": "Nội dung.",
    }
    other_part = {**base, "part": "2.6"}
    other_text = {**base, "text": "Nội dung khác."}

    assert stable_chunk_id(base) == stable_chunk_id(dict(base))
    assert stable_chunk_id(base) != stable_chunk_id(other_part)
    assert stable_chunk_id(base) != stable_chunk_id(other_text)


def test_dataset_loader_preserves_pending_labels() -> None:
    records = [
        {
            "question": "Câu đã kiểm chứng?",
            "relevant_chunk_ids": ["chunk-1"],
            "answerable": True,
            "category": "policy",
            "label_status": "verified",
        },
        {
            "question": "Câu chờ xác nhận?",
            "relevant_chunk_ids": [],
            "answerable": None,
            "category": "colloquial",
            "label_status": "pending",
        },
    ]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "evaluation.jsonl"
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        examples = load_evaluation_dataset(path)

    assert len(examples) == 2
    assert examples[1].label_status == "pending"
    assert examples[1].answerable is None


def test_metrics_exclude_pending_and_unanswerable_questions() -> None:
    relevant_chunk = {
        "part": "2.1",
        "article": 7,
        "clauses": [5],
        "page_start": 16,
        "text": "Tối đa 27 tín chỉ.",
    }
    relevant_id = stable_chunk_id(relevant_chunk)
    examples = [
        EvaluationExample(
            "Câu đúng?",
            (relevant_id,),
            True,
            "policy",
            "verified",
        ),
        EvaluationExample(
            "Câu ngoài tài liệu?",
            (),
            False,
            "out_of_scope",
            "verified",
        ),
        EvaluationExample(
            "Câu chờ xác nhận?",
            (),
            None,
            "colloquial",
            "pending",
        ),
    ]

    metrics, ranks = calculate_metrics(
        examples,
        lambda question: [relevant_chunk],
    )

    assert metrics.evaluated_questions == 1
    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert metrics.mrr == 1.0
    assert set(ranks) == {"Câu đúng?"}
