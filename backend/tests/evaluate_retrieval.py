import sys

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import ask


QUESTIONS = [
    "Sinh viên bị cảnh báo học tập khi nào?",
    "Khi nào sinh viên bị buộc thôi học?",
    "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ?",
    "Điều kiện để được xét tốt nghiệp là gì?",
    "Sinh viên được nhận học bổng khi nào?",
    "Điểm thấp quá thì có bị đuổi học không?",
]


def evaluate() -> None:
    chunks, vectorizer, document_vectors = load_index()

    print("TỔNG SỐ CHUNKS:", len(chunks))
    print("VOCABULARY:", len(vectorizer.vocabulary))

    for index, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 65)
        print(f"CÂU HỎI {index}: {question}")

        results = ask(
            question,
            chunks,
            vectorizer,
            document_vectors,
        )
        for rank, result in enumerate(results, start=1):
            print(f"\nTOP {rank}")
            print(f"Điều: {result.get('article')}")
            print(f"Khoản: {result.get('clauses', [])}")
            print(f"Trang: {result.get('page_start')}")
            print(f"Score: {result['score']:.4f}")
            print(f"Nội dung: {result['text']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    evaluate()
