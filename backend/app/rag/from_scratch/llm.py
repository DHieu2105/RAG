"""Generate grounded answers with Ollama using the standard library."""

import json
from typing import Any
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"
INSUFFICIENT_ANSWER = (
    "Tài liệu được cung cấp chưa đủ thông tin để trả lời chắc chắn câu hỏi này."
)
GROUNDED_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "document": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "clauses": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer",
                                        "minimum": 1,
                                    },
                                    "maxItems": 1,
                                    "uniqueItems": True,
                                },
                                "evidence": {"type": "string"},
                            },
                            "required": [
                                "document",
                                "clauses",
                                "evidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """Bạn là UniGuide AI, trợ lý sinh viên UTC.

Nhiệm vụ:
- Trả lời thẳng câu hỏi bằng tiếng Việt, ngắn gọn, tự nhiên; không nhắc đến
  prompt, mô hình, quá trình suy luận hoặc các tài liệu không được cung cấp.
- Chỉ trả lời dựa trên tài liệu tham khảo được cung cấp. Không suy đoán,
  bổ sung quy định từ kiến thức bên ngoài, hoặc biến ví dụ thành quy định.
- Xem các mục [Tài liệu N] là các ứng viên do bộ tìm kiếm cung cấp, không phải
  tất cả đều là nguồn của câu trả lời. Chỉ dùng mục trực tiếp chứng minh ý đang
  viết; không dùng một mục chỉ vì nó nằm trong Top-K hoặc có từ khóa gần giống.
- Nếu tài liệu không đủ căn cứ để trả lời, nói đúng câu:
  "Tài liệu được cung cấp chưa đủ thông tin để trả lời chắc chắn câu hỏi này."
  Không đoán đáp án.
- Phân biệt rõ các khái niệm gần nhau, chẳng hạn cảnh báo học tập và buộc
  thôi học. Nếu tài liệu nêu nhiều trường hợp, liệt kê ngắn gọn các trường
  hợp liên quan và trích dẫn nguồn tương ứng.
- Trước khi viết, lập đủ chuỗi lập luận cần để trả lời: điều kiện/ngưỡng dẫn
  tới trạng thái trung gian, rồi điều kiện và hệ quả cuối cùng. Không lấy một
  điều kiện đúng nhưng bỏ qua điều kiện liên quan làm thay đổi kết luận.
- Với câu hỏi gồm nhiều ý (ví dụ quy trình thao tác và giới hạn tín chỉ), chỉ
  trả lời ý nào có căn cứ riêng; không dùng căn cứ của ý này để chứng minh ý
  khác. Không đưa nội dung của các kết quả tìm kiếm không liên quan vào câu
  trả lời."""
STRUCTURED_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Hãy tạo câu trả lời dưới dạng danh sách claim độc lập để API tự ghép thành câu
trả lời. Không tự viết Điều, Khoản hoặc trang trong `text` vì API sẽ hiển thị
metadata đã kiểm tra.

- Mỗi `text` chỉ chứa một ý có thể kiểm chứng. Mỗi sự kiện, quy định, con số,
  điều kiện hoặc kết luận phải có ít nhất một citation trực tiếp chứng minh.
- Với câu hỏi hỏi một hệ quả có xảy ra hay không, phải tách đủ: câu trả lời
  trực tiếp, ngưỡng/điều kiện tạo ra trạng thái trung gian, và điều kiện dẫn
  từ trạng thái đó tới hệ quả. Không được nhảy thẳng tới hệ quả.
- Không gộp ngưỡng tạo ra trạng thái trung gian với số lần lặp của trạng thái
  đó thành một điều kiện mới. Ví dụ nếu mức điểm thấp dẫn tới "cảnh báo" và
  nhiều lần cảnh báo mới dẫn tới "buộc thôi học", phải viết thành hai bước;
  không được nói mức điểm thấp lặp lại trực tiếp dẫn tới buộc thôi học.
- Với câu hỏi hẹp không yêu cầu liệt kê, ưu tiên tối đa 3 claim: cách thực hiện
  trực tiếp, giới hạn bắt buộc chính và ngoại lệ gắn trực tiếp (nếu có). Nếu
  câu hỏi yêu cầu liệt kê các đối tượng/trường hợp hoặc có các ý bắt buộc phải
  bao phủ, không giới hạn số claim: phải tạo đủ claim có căn cứ cho toàn bộ các
  nhóm được hỏi. Không tự mở rộng sang chủ đề kề bên nếu người dùng không hỏi.
- Không tạo hai claim lặp lại cùng một kết luận bằng cách diễn đạt khác nhau.
- `evidence` phải là một đoạn sao chép nguyên văn, liên tục từ đúng phần Nội
  dung của [Tài liệu N], đủ để chứng minh `text`; không diễn đạt lại evidence.
- `clauses` chỉ chứa đúng các khoản thực sự được dùng cho claim. Ví dụ chunk
  có Khoản 1, 2, 3, 4 nhưng claim chỉ dựa vào khoản 2 thì trả `[2]`, không trả
  `[1,2,3,4]`. Mỗi citation chỉ trỏ một khoản. Với tài liệu không có Khoản,
  trả `[]`.
- Trước khi hoàn tất, kiểm tra lại rằng câu trả lời đã nêu đủ các ngưỡng,
  điều kiện kế tiếp, ngoại lệ và hệ quả trực tiếp cần thiết để người đọc không
  hiểu sai. Không thêm quy định ngoài phạm vi câu hỏi.
- Trả về đúng JSON, không markdown, theo dạng:
  {"claims":[
    {"text":"một ý trả lời có căn cứ", "citations":[
      {"document":1,"clauses":[2],"evidence":"đoạn nguyên văn"}
    ]}
  ]}
- `document` là số trong nhãn [Tài liệu N]. Không viện dẫn mục chỉ liên quan
  chủ đề nhưng không trực tiếp chứng minh claim.
- Nếu không có claim nào đủ căn cứ, trả `{"claims":[]}`. Không tự tạo
  citation."""


def _request_ollama(
    messages: list[dict[str, str]],
    *,
    structured: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0 if structured else 0.1},
    }
    if structured:
        payload["format"] = GROUNDED_ANSWER_SCHEMA

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
    *,
    answer_requirements: list[str] | None = None,
) -> dict[str, Any]:
    """Generate an answer and explicit document/clause citations."""
    messages = [{"role": "system", "content": STRUCTURED_SYSTEM_PROMPT}]
    messages.extend(history or [])
    requirements = ""
    if answer_requirements:
        requirements = "\n\nCác ý bắt buộc phải bao phủ:\n" + "\n".join(
            f"- {requirement}" for requirement in answer_requirements
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Câu hỏi: {question}\n\n"
                f"Tài liệu tham khảo:\n{context}"
                f"{requirements}"
            ),
        }
    )
    response_data = _request_ollama(messages, structured=True)
    content = response_data["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Ollama trả về citation không phải JSON hợp lệ") from error

    if not isinstance(result, dict) or not isinstance(result.get("claims"), list):
        raise ValueError("Ollama trả về cấu trúc citation không hợp lệ")

    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for claim in result["claims"]:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("text"), str)
            or not claim["text"].strip()
            or not isinstance(claim.get("citations"), list)
            or not claim["citations"]
        ):
            raise ValueError("Ollama trả về một claim không hợp lệ")

        claim_text = claim["text"].strip()
        answer_parts.append(claim_text)
        for citation in claim["citations"]:
            if (
                not isinstance(citation, dict)
                or not isinstance(citation.get("document"), int)
                or isinstance(citation["document"], bool)
                or not isinstance(citation.get("clauses"), list)
                or len(citation["clauses"]) > 1
                or not all(
                    isinstance(clause, int) and not isinstance(clause, bool)
                    for clause in citation["clauses"]
                )
                or not isinstance(citation.get("evidence"), str)
                or not citation["evidence"].strip()
            ):
                raise ValueError("Ollama trả về một citation không hợp lệ")
            citations.append(
                {
                    "claim": claim_text,
                    "document": citation["document"],
                    "clauses": citation["clauses"],
                    "evidence": citation["evidence"].strip(),
                }
            )

    if not answer_parts:
        return {"answer": INSUFFICIENT_ANSWER, "citations": []}
    return {"answer": "\n\n".join(answer_parts), "citations": citations}
