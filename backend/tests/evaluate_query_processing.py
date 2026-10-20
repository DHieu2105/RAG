"""Compare baseline TF-IDF retrieval with deterministic query processing."""

import sys

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import ask, ask_with_query_processing
from app.rag.from_scratch.query_processing import generate_search_queries

from .evaluate_retrieval import QUESTIONS


def _page_range(result: dict) -> str:
    page_start = result.get("page_start")
    page_end = result.get("page_end")
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _print_results(label: str, results: list[dict]) -> None:
    print(f"\n--- {label} ---")
    for rank, result in enumerate(results, start=1):
        print(f"\nTOP {rank}")
        print(f"PHẦN: {result.get('part')}")
        print(f"CHƯƠNG: {result.get('chapter')}")
        print(f"Điều: {result.get('article')}")
        print(f"Khoản: {result.get('clauses', [])}")
        print(f"Trang: {_page_range(result)}")
        print(f"Score: {result['score']:.6f}")
        if "tfidf_score" in result:
            print(f"Best TF-IDF Score: {result['tfidf_score']:.6f}")
            print(f"Matched Queries: {result['matched_queries']}")
        print(f"Nội dung: {result['text']}")


def evaluate() -> None:
    chunks, vectorizer, document_vectors = load_index()

    print("TỔNG SỐ CHUNKS:", len(chunks))
    print("VOCABULARY:", len(vectorizer.vocabulary))

    for index, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 65)
        print(f"CÂU HỎI {index}: {question}")
        print("TRUY VẤN TÌM KIẾM:")
        for query_index, query in enumerate(
            generate_search_queries(question),
            start=1,
        ):
            print(f"  {query_index}. {query}")

        baseline = ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
        )
        expanded = ask_with_query_processing(
            question,
            chunks,
            vectorizer,
            document_vectors,
        )
        _print_results("A. TF-IDF GỐC", baseline)
        _print_results("B. TF-IDF + QUERY PROCESSING", expanded)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
