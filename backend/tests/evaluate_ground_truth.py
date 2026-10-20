"""Evaluate four retrieval configurations against verified chunk labels."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from typing import Callable

from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.evaluation import (
    EvaluationExample,
    calculate_metrics,
    first_relevant_rank,
    load_evaluation_dataset,
    stable_chunk_id,
)
from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import (
    ask,
    ask_bm25_with_query_processing,
    ask_hybrid,
    ask_with_query_processing,
)
from app.rag.from_scratch.query_processing import generate_search_queries
from app.rag.from_scratch.reranker import (
    ANSWER_CUE_WEIGHT,
    COVERAGE_WEIGHT,
    RANK_OFFSET,
    RANK_WEIGHT,
    lexical_terms,
)


DATASET_PATH = Path(__file__).parent / "data" / "retrieval_evaluation.jsonl"


def _rank_value(rank: int | None) -> int:
    return rank if rank is not None else 10**9


def _chunk_label(chunk: dict) -> str:
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    pages = str(page_start) if page_end in (None, page_start) else f"{page_start}-{page_end}"
    return (
        f"{stable_chunk_id(chunk)} | phần={chunk.get('part')} "
        f"điều={chunk.get('article')} khoản={chunk.get('clauses', [])} "
        f"trang={pages}"
    )


def _build_retrievers() -> tuple[
    list[dict],
    dict[str, Callable[[str], list[dict]]],
]:
    chunks, vectorizer, document_vectors = load_index()
    bm25_index = build_bm25_index(chunks)
    retrievers: dict[str, Callable[[str], list[dict]]] = {
        "A. TF-IDF gốc": lambda question: ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
        ),
        "B. TF-IDF + Query Processing": lambda question: (
            ask_with_query_processing(
                question,
                chunks,
                vectorizer,
                document_vectors,
            )
        ),
        "C. BM25 + Query Processing": lambda question: (
            ask_bm25_with_query_processing(
                question,
                chunks,
                bm25_index,
            )
        ),
        "D. Hybrid + Query Processing + Reranking": lambda question: (
            ask_hybrid(
                question,
                chunks,
                vectorizer,
                document_vectors,
                bm25_index,
            )
        ),
    }
    return chunks, retrievers


def _validate_relevant_ids(
    examples: list[EvaluationExample],
    chunks: list[dict],
) -> None:
    known_ids = {stable_chunk_id(chunk) for chunk in chunks}
    missing = {
        chunk_id
        for example in examples
        if example.label_status == "verified"
        for chunk_id in example.relevant_chunk_ids
        if chunk_id not in known_ids
    }
    if missing:
        raise ValueError(
            "Ground truth chứa chunk ID không có trong index: "
            + ", ".join(sorted(missing))
        )


def _metrics_section(
    title: str,
    examples: list[EvaluationExample],
    cached_results: dict[str, dict[str, list[dict]]],
) -> dict[str, dict[str, int | None]]:
    print(f"\n## {title}")
    print("| Cấu hình | Số câu | Hit@1 | Hit@5 | MRR |")
    print("|---|---:|---:|---:|---:|")
    all_ranks: dict[str, dict[str, int | None]] = {}
    for name, results_by_question in cached_results.items():
        metrics, ranks = calculate_metrics(
            examples,
            lambda question, cache=results_by_question: cache[question],
        )
        all_ranks[name] = ranks
        print(
            f"| {name} | {metrics.evaluated_questions} | "
            f"{metrics.hit_at_1:.3f} | {metrics.hit_at_5:.3f} | "
            f"{metrics.mrr:.3f} |"
        )
    print(
        "\nMRR được tính trên danh sách 5 kết quả mà cấu hình thực tế trả về; "
        "không tìm thấy trong Top-5 nhận reciprocal rank bằng 0."
    )
    return all_ranks


def _change_section(
    examples: list[EvaluationExample],
    ranks: dict[str, dict[str, int | None]],
) -> None:
    print("\n## Câu cải thiện hoặc giảm so với A")
    eligible = [
        example
        for example in examples
        if example.label_status == "verified" and example.answerable is True
    ]
    baseline_name = "A. TF-IDF gốc"
    baseline = ranks[baseline_name]
    for name, current in ranks.items():
        if name == baseline_name:
            continue
        improved: list[str] = []
        degraded: list[str] = []
        unchanged = 0
        for example in eligible:
            old_rank = baseline[example.question]
            new_rank = current[example.question]
            detail = f"{example.question} ({old_rank or '-'} → {new_rank or '-'})"
            if _rank_value(new_rank) < _rank_value(old_rank):
                improved.append(detail)
            elif _rank_value(new_rank) > _rank_value(old_rank):
                degraded.append(detail)
            else:
                unchanged += 1

        print(f"\n### {name}")
        print(f"- Cải thiện: {len(improved)}")
        for detail in improved:
            print(f"  - {detail}")
        print(f"- Giảm: {len(degraded)}")
        for detail in degraded:
            print(f"  - {detail}")
        print(f"- Không đổi: {unchanged}")


def _noise_section(
    examples: list[EvaluationExample],
    cached_results: dict[str, dict[str, list[dict]]],
) -> None:
    print("\n## Kết quả Top-1 nhiễu trên câu có đáp án")
    eligible = [
        example
        for example in examples
        if example.label_status == "verified" and example.answerable is True
    ]
    for name, results_by_question in cached_results.items():
        noisy: list[tuple[EvaluationExample, dict, list[str]]] = []
        for example in eligible:
            results = results_by_question[example.question]
            if not results:
                continue
            if first_relevant_rank(results[:1], example.relevant_chunk_ids) == 1:
                continue
            top = results[0]
            overlap = sorted(
                lexical_terms(example.question)
                & lexical_terms(top.get("text", ""))
            )
            noisy.append((example, top, overlap))

        print(f"\n### {name}: {len(noisy)}/{len(eligible)} câu có Top-1 nhiễu")
        for example, top, overlap in noisy:
            overlap_text = ", ".join(overlap) if overlap else "không có"
            print(f"- {example.question}")
            print(f"  - Top-1: {_chunk_label(top)}")
            print(f"  - Từ khóa giao nhau: {overlap_text}")
            print("  - Kết luận: có thể liên quan từ vựng nhưng không thuộc nhãn evidence trực tiếp.")


def _answerability_section(
    examples: list[EvaluationExample],
    cached_results: dict[str, dict[str, list[dict]]],
) -> None:
    print("\n## Câu hỏi không đủ căn cứ")
    unanswerable = [
        example
        for example in examples
        if example.label_status == "verified" and example.answerable is False
    ]
    for example in unanswerable:
        print(f"\n### {example.question}")
        print(f"- Nhãn: không có evidence trực tiếp trong corpus. {example.notes}")
        for name, results_by_question in cached_results.items():
            results = results_by_question[example.question]
            if not results:
                print(f"- {name}: không trả kết quả")
                continue
            top = results[0]
            print(
                f"- {name}: vẫn trả Top-1 score={top.get('score', 0.0):.6f}; "
                f"{_chunk_label(top)}"
            )
    print(
        "\nĐiểm cosine/BM25/RRF/rerank dương chỉ thể hiện mức khớp tương đối "
        "trong tập ứng viên, không chứng minh passage trả lời được câu hỏi."
    )


def _reranker_section(
    examples: list[EvaluationExample],
    cached_results: dict[str, dict[str, list[dict]]],
) -> None:
    hybrid_name = "D. Hybrid + Query Processing + Reranking"
    hybrid_results = cached_results[hybrid_name]
    expanded = sum(
        len(generate_search_queries(example.question)) > 1
        for example in examples
    )
    reranked = sum(
        bool(hybrid_results[example.question])
        and "rerank_score" in hybrid_results[example.question][0]
        for example in examples
    )
    print("\n## Cơ chế reranking hiện tại")
    print(
        f"- Công thức: {RANK_WEIGHT} × 1/({RANK_OFFSET} + rank) + "
        f"{COVERAGE_WEIGHT} × query_coverage + "
        f"{ANSWER_CUE_WEIGHT} × answer_cue."
    )
    print(
        f"- Query expansion kích hoạt ở {expanded}/{len(examples)} câu; "
        f"rerank thực sự xuất hiện ở {reranked}/{len(examples)} câu."
    )
    print(
        "- query_coverage chỉ đếm giao từ sau stop-word; answer_cue chỉ nhận "
        "một nhóm mẫu hỏi điều kiện. Vì vậy passage lặp đúng từ khóa nhưng chỉ "
        "nói thủ tục/hậu quả vẫn có thể vượt passage trả lời trực tiếp."
    )
    print(
        "- Rank prior có trọng số lớn nên reranker chủ yếu bảo toàn thứ tự RRF; "
        "nó chưa kiểm tra quan hệ entailment giữa câu hỏi và một câu evidence."
    )


def evaluate() -> None:
    examples = load_evaluation_dataset(DATASET_PATH)
    chunks, retrievers = _build_retrievers()
    _validate_relevant_ids(examples, chunks)

    cached_results = {
        name: {
            example.question: retrieve(example.question)
            for example in examples
        }
        for name, retrieve in retrievers.items()
    }

    status_counts = Counter(example.label_status for example in examples)
    answerability_counts = Counter(
        str(example.answerable)
        for example in examples
        if example.label_status == "verified"
    )
    print("# ĐÁNH GIÁ RETRIEVAL CÓ GROUND TRUTH")
    print(f"- Dataset: {DATASET_PATH}")
    print(f"- Tổng câu: {len(examples)}")
    print(f"- Nhãn verified: {status_counts['verified']}")
    print(f"- Nhãn pending: {status_counts['pending']}")
    print(f"- Verified answerable: {answerability_counts['True']}")
    print(f"- Verified unanswerable: {answerability_counts['False']}")
    print(f"- Số chunk trong index: {len(chunks)}")

    all_ranks = _metrics_section(
        "Toàn bộ câu answerable có nhãn verified",
        examples,
        cached_results,
    )
    colloquial = [
        example for example in examples if example.category == "colloquial"
    ]
    _metrics_section(
        "Báo cáo riêng: câu hỏi đời thường",
        colloquial,
        cached_results,
    )
    _change_section(examples, all_ranks)
    _noise_section(examples, cached_results)
    _answerability_section(examples, cached_results)
    _reranker_section(examples, cached_results)

    print("\n## Nhãn còn chờ xác nhận")
    for example in examples:
        if example.label_status == "pending":
            print(f"- {example.question}: {example.notes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 path for the generated text report.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.output is None:
        evaluate()
        return

    buffer = StringIO()
    with redirect_stdout(buffer):
        evaluate()
    report = buffer.getvalue()
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
