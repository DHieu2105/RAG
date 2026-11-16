"""Generate answers with Google Gemini API using the standard library."""

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def _ensure_env() -> None:
    backend_env = Path(__file__).resolve().parents[3] / ".env"
    if backend_env.exists():
        load_dotenv(backend_env, override=False)
    load_dotenv(override=False)


_ensure_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

SYSTEM_PROMPT = """Bạn là trợ lý học vụ thông minh hỗ trợ sinh viên giải đáp quy chế, quy định từ Sổ tay sinh viên.

Quy tắc trả lời BẮT BUỘC:
1. TRẢ LỜI NGẮN GỌN, SÚC TÍCH, ĐI THẲNG VÀO TRỌNG TÂM CÂU HỎI. Tuyệt đối không mở bài, chào hỏi hay giải thích lan man dài dòng.
2. Trình bày bằng các gạch đầu dòng rõ ràng (- ...). Mỗi ý bắt buộc phải trên một dòng riêng biệt, không viết liền khối.
3. ĐIỀU KHOẢN TRÍCH DẪN: Ở cuối mỗi gạch đầu dòng, BẮT BUỘC ghi rõ căn cứ quy định trong ngoặc đơn dựa trên metadata của [Tài liệu tham khảo], ví dụ: *(Khoản 1 Điều 15, Trang 25)* hoặc *(Điều 11, Trang 23)* hoặc *(Phần 2.7, Trang 50)*.
4. Chỉ sử dụng thông tin có trong [Tài liệu tham khảo]. Giữ nguyên các con số, ngưỡng điểm, số tín chỉ, mốc thời gian và điều kiện bắt buộc. Tuyệt đối không suy đoán hay thêm thắt ngoài tài liệu.
5. Nếu tài liệu tham khảo hoàn toàn không có thông tin liên quan hoặc không thể trả lời, hãy trả lời chính xác:
   "Không tìm thấy thông tin liên quan trong sổ tay sinh viên."
"""


def _request_gemini(
    system_prompt: str,
    prompt_text: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Send request to Google Gemini API using urllib with transient retry."""
    _ensure_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    model = os.getenv("GEMINI_MODEL", "").strip() or GEMINI_MODEL or "gemini-3.5-flash-lite"
    if not api_key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY trong .env")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    contents: list[dict[str, Any]] = []
    if history:
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}],
            })

    contents.append({
        "role": "user",
        "parts": [{"text": prompt_text}],
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=90) as response:
                response_data = json.load(response)
            break
        except Exception as exc:
            is_transient = False
            if isinstance(exc, HTTPError) and exc.code in (429, 500, 502, 503, 504):
                is_transient = True
            elif isinstance(exc, URLError):
                is_transient = True
            if is_transient and attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            error_msg = f"Lỗi gọi Gemini API: {exc}"
            if hasattr(exc, "read"):
                try:
                    error_msg += f" - {exc.read().decode('utf-8')}"
                except Exception:
                    pass
            print(f"[Gemini Error] {error_msg}")
            raise RuntimeError(error_msg) from exc

    candidates = response_data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()

    return "Không nhận được phản hồi từ Gemini API."


def generate_answer(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Ask Google Gemini API to answer a question using only the supplied context."""
    prompt_text = (
        f"Câu hỏi: {question}\n\n"
        f"[Tài liệu tham khảo]:\n{context}"
    )
    return _request_gemini(SYSTEM_PROMPT, prompt_text, history=history)
