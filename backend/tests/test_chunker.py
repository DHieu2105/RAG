from app.rag.chunker import chunk_text

text = "Sinh viên đăng ký học phần theo quy định của nhà trường. " * 30

chunks = chunk_text(
    text,
    chunk_size=200,
    chunk_overlap=30
)

print("Tổng số chunks:", len(chunks))

for index, chunk in enumerate(chunks):
    print(f"\n--- Chunk {index + 1} ---")
    print("Số ký tự:", len(chunk))
    print(chunk)