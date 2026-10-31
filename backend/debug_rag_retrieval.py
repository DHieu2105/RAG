import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.pipeline import (
    ask,
    ask_bm25,
    ask_hybrid,
    retrieve_for_chat,
)
from app.rag.from_scratch.query_processing import generate_search_queries
from app.rag.from_scratch.query_understanding import (
    analyze_query,
    lexical_terms,
)
from app.rag.from_scratch.grounding import context_is_relevant
from app.rag.from_scratch.tokenizer import tokenize

QUESTION = "Ai được miễn giảm học phí?"


def show_results(label, results, limit=5):
    print(f"\n--- {label} ---")
    if not results:
        print("Không có kết quả.")
        return
    for i, item in enumerate(results[:limit], 1):
        score = item.get("score", 0.0)
        print(
            f"\n{i}. Score: {score:.4f}"
            f" | Điều {item.get('article')}"
            f" | {item.get('article_title')}"
            f" | Khoản {item.get('clauses')}"
            f" | Trang {item.get('page_start')}"
        )
        content = " ".join(str(item.get("text") or "").split())
        print(content[:350])


print("Đang tải index...")
chunks, vectorizer, vectors = load_index()

print("Đang tạo BM25...")
bm25 = build_bm25_index(chunks)

analysis = analyze_query(QUESTION)
queries = generate_search_queries(QUESTION)

print("\n========== PHÂN TÍCH CÂU HỎI ==========")
print("Question:", QUESTION)
print("Focus terms:", analysis.focus_terms)
print("Intents:", analysis.intents)
print("Search queries:", queries)

vocabulary_terms = set()
for token in vectorizer.vocabulary:
    vocabulary_terms.update(lexical_terms(token))

print(
    "Focus terms không có trong vocabulary:",
    sorted(set(analysis.focus_terms) - vocabulary_terms),
)

print("\n========== TOKENIZER ==========")
for query in queries:
    print(f"Query: '{query}'")
    print("  tokenize:     ", tokenize(query))
    print("  lexical_terms:", list(lexical_terms(query)))

print("\n========== TRUY XUẤT TỪNG QUERY ==========")
for q in queries:
    res_tfidf = ask(q, chunks, vectorizer, vectors, top_k=3)
    show_results(f"TF-IDF: '{q}'", res_tfidf, limit=3)
    res_bm25 = ask_bm25(q, chunks, bm25, top_k=3)
    show_results(f"BM25: '{q}'", res_bm25, limit=3)

print("\n========== HYBRID RETRIEVAL ==========")
hybrid_results = ask_hybrid(QUESTION, chunks, vectorizer, vectors, bm25, top_k=5)
show_results("Hybrid Retrieval", hybrid_results, limit=5)

print("\n========== SAU KHI XỬ LÝ CẤU TRÚC ==========")
chat_results = retrieve_for_chat(
    QUESTION,
    chunks,
    vectorizer,
    vectors,
    bm25_index=bm25,
    retrieval_mode="hybrid",
)
show_results("Sau retrieve_for_chat", chat_results, limit=5)

known_terms = set(vectorizer.vocabulary)
is_relevant = context_is_relevant(analysis, chat_results, known_terms=known_terms)
print(f"\n=> context_is_relevant: {is_relevant}")

focus = set(analysis.focus_terms)
title_terms = set()
for chunk in chat_results:
    title = str(chunk.get("article_title") or "")
    title_terms.update(lexical_terms(title))

normalized_vocabulary = set()
for term in known_terms:
    normalized_vocabulary.update(lexical_terms(term))
searchable_terms = normalized_vocabulary | title_terms
missing_terms = focus - searchable_terms
print(f"Missing focus terms from searchable: {missing_terms}")

for i, chunk in enumerate(chat_results, 1):
    title = str(chunk.get("article_title") or "")
    content = str(chunk.get("text") or "")
    terms = set(lexical_terms(f"{title} {content}"))
    cov = len(terms & focus) / len(focus) if focus else 0.0
    print(
        f"Chunk {i} (Điều {chunk.get('article')} - Khoản {chunk.get('clauses')}) coverage: {cov:.2f} (terms matching: {terms & focus})"
    )

