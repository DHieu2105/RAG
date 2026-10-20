"""Generate grounded answers with Ollama using the standard library."""

import json
from typing import Any
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"
SYSTEM_PROMPT = """Bạn là UniGuide AI, trợ lý sinh viên UTC.

Nhiệm vụ:
- Trả lời thẳng câu hỏi bằng tiếng Việt, ngắn gọn, tự nhiên; không nhắc đến
  prompt, mô hình, quá trình suy luận hoặc các tài liệu không được cung cấp.
- Chỉ trả lời dựa trên tài liệu tham khảo được cung cấp. Không suy đoán,
  bổ sung quy định từ kiến thức bên ngoài, hoặc biến ví dụ thành quy định.
- Với mỗi quy định hoặc kết luận quan trọng, trích dẫn ngay sau câu trả lời
  bằng đúng metadata xuất hiện trong tài liệu, theo mẫu:
  (Phần ..., Chương ..., Điều ..., khoản ..., trang PDF ...)
  Chỉ đưa các thành phần có trong tài liệu; Không tự tạo số Điều, Khoản,
  Phần, Chương hoặc trang. Có thể dùng khoảng trang, ví dụ trang PDF 45-46.
- Nếu tài liệu không đủ căn cứ để trả lời, nói rõ:
  "Tài liệu được cung cấp chưa đủ thông tin để trả lời chắc chắn câu hỏi
  này." Không đoán đáp án.
- Phân biệt rõ các khái niệm gần nhau, chẳng hạn cảnh báo học tập và buộc
  thôi học. Nếu tài liệu nêu nhiều trường hợp, liệt kê ngắn gọn các trường
  hợp liên quan và trích dẫn nguồn tương ứng.
- Không dùng trích dẫn chung chung như "theo tài liệu"; mọi trích dẫn phải
  lấy từ metadata của một hoặc nhiều mục [Tài liệu N]."""
STRUCTURED_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Trước khi trả lời, hãy kiểm tra tính đầy đủ của câu trả lời:
- Xác định tất cả điều kiện, ngưỡng, ngoại lệ và hệ quả cần thiết để trả lời
  câu hỏi; không dừng ở một điều kiện đúng nếu tài liệu còn nêu điều kiện
  liên quan.
- Mỗi ý chính phải có căn cứ từ một mục [Tài liệu N] và các khoản cụ thể.
- Trả về đúng JSON, không markdown, theo dạng:
  {"answer":"...", "citations":[
    {"document":1,"clauses":[1,2]}
  ]}
- `document` là số trong nhãn [Tài liệu N]. `clauses` chỉ chứa các khoản mà
  câu trả lời thực sự sử dụng; không liệt kê toàn bộ khoản của tài liệu.
- Nếu câu trả lời không đủ căn cứ, vẫn trả về JSON và để citations là [].
  Không tự tạo citation."""


def _request_ollama(
    messages: list[dict[str, str]],
    *,
    structured: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if structured:
        payload["format"] = "json"

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        response_data = json.load(response)
    return response_data


def generate_answer(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Ask Ollama to answer a question using only the supplied context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    response_data = _request_ollama(messages)
    return response_data["message"]["content"]


def generate_grounded_answer(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate an answer and explicit document/clause citations."""
    messages = [{"role": "system", "content": STRUCTURED_SYSTEM_PROMPT}]
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
    response_data = _request_ollama(messages, structured=True)
    content = response_data["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Ollama trả về citation không phải JSON hợp lệ") from error

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("answer"), str)
        or not isinstance(result.get("citations"), list)
    ):
        raise ValueError("Ollama trả về cấu trúc citation không hợp lệ")

    return result
