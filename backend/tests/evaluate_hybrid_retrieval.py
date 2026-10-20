"""Compare four lexical retrieval configurations without calling Ollama."""

import sys

from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import (
    ask,
    ask_bm25_with_query_processing,
    ask_hybrid,
    ask_with_query_processing,
)
from app.rag.from_scratch.query_processing import generate_search_queries

from .evaluate_retrieval import QUESTIONS as CORE_QUESTIONS


EXPLORATORY_QUESTIONS = [
    "Mỗi học kỳ chính phải đăng ký ít nhất bao nhiêu tín chỉ?",
    "Có được rút bớt học phần đã đăng ký không?",
    "Điểm F được xác định như thế nào?",
    "Điểm trung bình tích lũy bao nhiêu thì đủ điều kiện tốt nghiệp?",
    "Sinh viên được nghỉ học tạm thời trong trường hợp nào?",
    "Thời gian tối đa để hoàn thành chương trình là bao lâu?",
    "Sinh viên có thể học cùng lúc hai chương trình không?",
    "Muốn chuyển ngành cần đáp ứng điều kiện gì?",
    "Thi hộ bị xử lý như thế nào?",
    "Dùng văn bằng giả để nhập học sẽ bị xử lý ra sao?",
    "Điểm rèn luyện dưới 50 được xếp loại gì?",
    "Bị cảnh cáo thì kết quả rèn luyện tối đa loại nào?",
    "Học bổng khuyến khích học tập được cấp mấy tháng?",
    "Sinh viên cử tuyển có được hưởng học bổng chính sách không?",
    "Những đối tượng sinh viên nào được miễn học phí?",
    "Sinh viên ngoại trú có những trách nhiệm gì?",
    "Quên mật khẩu tài khoản quản lý đào tạo thì làm thế nào?",
    "Đăng ký phòng ký túc xá trên hệ thống nào?",
    "Trường có cấp laptop miễn phí cho sinh viên không?",
    "Có xe buýt riêng đưa sinh viên về quê không?",
]


def _score_type(result: dict) -> str:
    if "rerank_score" in result:
        return "Lexical rerank score"
    if "hybrid_source_score" in result:
        return "Hybrid RRF score"
    if "bm25_score" in result:
        return "Query-fusion RRF score; source BM25"
    if "tfidf_score" in result:
        return "Query-fusion RRF score; source cosine"
    return "Native retriever score"


def _page_range(result: dict) -> str:
    page_start = result.get("page_start")
    page_end = result.get("page_end")
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _print_results(label: str, results: list[dict]) -> None:
    print(f"\n--- {label} ---")
    for rank, result in enumerate(results, start=1):
        print(f"TOP {rank}")
        print(f"PHẦN: {result.get('part')}")
        print(f"CHƯƠNG: {result.get('chapter')}")
        print(f"Điều: {result.get('article')}")
        print(f"Khoản: {result.get('clauses', [])}")
        print(f"Trang: {_page_range(result)}")
        print(f"Score type: {_score_type(result)}")
        print(f"Score: {result['score']:.6f}")
        if "retrieval_score" in result:
            print(f"Pre-rerank score: {result['retrieval_score']:.6f}")
        excerpt = " ".join(result["text"].split())
        print(f"Nội dung: {excerpt[:500]}")
        print()


def _evaluate_group(
    group_name: str,
    questions: list[str],
    chunks: list[dict],
    vectorizer,
    document_vectors: list[list[float]],
    bm25_index,
) -> None:
    print("\n" + "#" * 72)
    print(group_name)

    for index, question in enumerate(questions, start=1):
        print("\n" + "=" * 72)
        print(f"CÂU HỎI {index}: {question}")
        print("TRUY VẤN:")
        for query in generate_search_queries(question):
            print(f"- {query}")

        configurations = (
            (
                "A. TF-IDF GỐC",
                ask(question, chunks, vectorizer, document_vectors),
            ),
            (
                "B. TF-IDF + QUERY PROCESSING",
                ask_with_query_processing(
                    question,
                    chunks,
                    vectorizer,
                    document_vectors,
                ),
            ),
            (
                "C. BM25 + QUERY PROCESSING",
                ask_bm25_with_query_processing(
                    question,
                    chunks,
                    bm25_index,
                ),
            ),
            (
                "D. HYBRID + QUERY PROCESSING + RERANKING",
                ask_hybrid(
                    question,
                    chunks,
                    vectorizer,
                    document_vectors,
                    bm25_index,
                ),
            ),
        )
        for label, results in configurations:
            _print_results(label, results)


def evaluate() -> None:
    chunks, vectorizer, document_vectors = load_index()
    bm25_index = build_bm25_index(chunks)

    print("TỔNG SỐ CHUNKS:", len(chunks))
    print("VOCABULARY:", len(vectorizer.vocabulary))
    print("LƯU Ý: Score của bốn cấu hình thuộc các thang đo khác nhau.")
    _evaluate_group(
        "BỘ 6 CÂU HIỆN TẠI",
        CORE_QUESTIONS,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index,
    )
    _evaluate_group(
        "20 CÂU THĂM DÒ CHƯA GÁN NHÃN",
        EXPLORATORY_QUESTIONS,
        chunks,
        vectorizer,
        document_vectors,
        bm25_index,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
