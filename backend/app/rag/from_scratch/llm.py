"""Generate grounded answers with Ollama using the standard library."""

import json
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"


def generate_answer(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Ask Ollama to answer a question using only the supplied context."""
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý sinh viên UTC. "
                "Chỉ trả lời dựa trên tài liệu được cung cấp. "
                "Nếu tài liệu không có câu trả lời, hãy nói rõ. "
                "Khi trả lời, hãy trích dẫn Điều và trang PDF."
            ),
        }
    ]
    messages.extend(history or [])
    messages.append(
        {
            "role": "user",
            "content": (
                f"Câu hỏi: {question}\n\n"
                f"Tài liệu tham khảo:\n{context}"
            ),
        }
    )
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        response_data = json.load(response)

    return response_data["message"]["content"]
