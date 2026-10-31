import sys
import time

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.pipeline import chat_with_sources

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("Đang tải RAG Index...")
chunks, vectorizer, vectors = load_index()

print("Đang khởi tạo BM25...")
bm25 = build_bm25_index(chunks)

print(f"Đã tải {len(chunks)} chunks")
print("UniGuide AI Local Test sẵn sàng!")
print("Gõ exit để thoát.")

while True:
    question = input("\nNhập câu hỏi: ").strip()
    print("DEBUG QUESTION:", repr(question))

    if question.lower() == "exit":
        break

    if not question:
        continue

    start = time.perf_counter()

    try:
        result = chat_with_sources(
            question,
            chunks,
            vectorizer,
            vectors,
            history=[],
            bm25_index=bm25,
            retrieval_mode="hybrid",
        )

        elapsed = time.perf_counter() - start

        print("\n" + "=" * 60)
        print("CÂU TRẢ LỜI:")
        print(result["answer"])

        print("\nNGUỒN DẪN:")
        sources = result.get("sources", [])

        if not sources:
            print("Không có nguồn dẫn hợp lệ.")
        else:
            for i, source in enumerate(sources, 1):
                print(
                    f"{i}. Phần {source.get('part')} | "
                    f"Điều {source.get('article')} | "
                    f"Khoản {source.get('clauses')} | "
                    f"Trang {source.get('page_start')}-"
                    f"{source.get('page_end')}"
                )

        print(f"\nThời gian xử lý: {elapsed:.2f} giây")
        print("=" * 60)

    except Exception as error:
        print(f"\nLỖI: {type(error).__name__}: {error}")
