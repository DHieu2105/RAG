from unittest.mock import patch

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.from_scratch.pipeline import chat


def test_chat_builds_context_and_generates_answer() -> None:
    chunks = [
        {
            "article": 11,
            "clauses": [1],
            "page_start": 23,
            "text": "Sinh viên bị cảnh báo học tập.",
        }
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunks[0]["text"]])
    vectors = vectorizer.transform([chunks[0]["text"]])

    with patch(
        "app.rag.from_scratch.pipeline.generate_answer",
        return_value="Câu trả lời từ Ollama.",
    ) as mocked_generate_answer:
        answer = chat(
            "Khi nào sinh viên bị cảnh báo học tập?",
            chunks,
            vectorizer,
            vectors,
        )

    assert answer == "Câu trả lời từ Ollama."
    mocked_generate_answer.assert_called_once()
    question, context = mocked_generate_answer.call_args.args
    assert question == "Khi nào sinh viên bị cảnh báo học tập?"
    assert "[Tài liệu 1]" in context
    assert "Điều: 11" in context
    assert "Khoản: 1" in context
    assert "Trang PDF: 23" in context
    assert "Sinh viên bị cảnh báo học tập." in context


def test_chat_skips_ollama_when_nothing_is_retrieved() -> None:
    chunks = [
        {
            "article": 11,
            "clauses": [1],
            "page_start": 23,
            "text": "Sinh viên bị cảnh báo học tập.",
        }
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([chunks[0]["text"]])
    vectors = vectorizer.transform([chunks[0]["text"]])

    with patch(
        "app.rag.from_scratch.pipeline.generate_answer"
    ) as mocked_generate_answer:
        answer = chat(
            "một câu hỏi không có token phù hợp",
            chunks,
            vectorizer,
            vectors,
        )

    assert answer == "Không tìm thấy thông tin liên quan trong sổ tay."
    mocked_generate_answer.assert_not_called()


if __name__ == "__main__":
    test_chat_builds_context_and_generates_answer()
    test_chat_skips_ollama_when_nothing_is_retrieved()
    print("All tests passed!")
