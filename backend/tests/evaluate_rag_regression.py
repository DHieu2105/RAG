"""Run the fixed end-to-end RAG regression set against local Ollama."""

import json
from pathlib import Path

from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import chat_with_sources, classify_answer_state
from app.rag.from_scratch.quality import evaluate_answer_response


DATASET_PATH = Path(__file__).parent / "data" / "rag_regression.jsonl"


def evaluate() -> None:
    examples = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks, vectorizer, vectors = load_index()
    bm25_index = build_bm25_index(chunks)
    metric_names = (
        "citation_correctness",
        "faithfulness",
        "completeness",
        "scope_accuracy",
        "abstention_accuracy",
    )
    totals = {name: 0 for name in metric_names}

    print("# RAG END-TO-END REGRESSION")
    for example in examples:
        response = chat_with_sources(
            example["question"],
            chunks,
            vectorizer,
            vectors,
            bm25_index=bm25_index,
        )
        quality = evaluate_answer_response(
            example["question"],
            answerable=example["answerable"],
            response=response,
        )
        for name in metric_names:
            totals[name] += int(getattr(quality, name))
        print(
            f"- [{example['category']}] {example['question']}\n"
            f"  state={classify_answer_state(response['answer']).value}; "
            + "; ".join(
                f"{name}={int(getattr(quality, name))}"
                for name in metric_names
            )
        )

    count = len(examples)
    print("\n## Tổng hợp")
    for name in metric_names:
        print(f"- {name}: {totals[name]}/{count} = {totals[name] / count:.3f}")


if __name__ == "__main__":
    evaluate()
