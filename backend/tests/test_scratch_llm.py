import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
import pytest

from app.rag.from_scratch.llm import (
    SYSTEM_PROMPT,
    generate_answer,
)


def test_generate_answer_gemini_success() -> None:
    gemini_response = BytesIO(
        json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "- Sinh viên bị cảnh báo kết quả học tập nếu điểm dưới 1,0. *(Khoản 1 Điều 11, Trang 23)*"
                                }
                            ]
                        }
                    }
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123", "GEMINI_MODEL": "gemini-3.5-flash-lite"}):
        with patch("app.rag.from_scratch.llm.urlopen", return_value=gemini_response) as mocked_urlopen:
            answer = generate_answer(
                "Khi nào sinh viên bị cảnh báo học tập?",
                "[Tài liệu 1]\nĐiều: 11\nKhoản: 1\nTrang PDF: 23",
            )

    assert answer == "- Sinh viên bị cảnh báo kết quả học tập nếu điểm dưới 1,0. *(Khoản 1 Điều 11, Trang 23)*"
    mocked_urlopen.assert_called_once()

    request = mocked_urlopen.call_args.args[0]
    assert "https://generativelanguage.googleapis.com" in request.full_url
    assert "key=test-key-123" in request.full_url
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"

    payload = json.loads(request.data.decode("utf-8"))
    assert payload["generationConfig"]["temperature"] == 0.1
    assert payload["generationConfig"]["maxOutputTokens"] == 2048
    assert payload["system_instruction"]["parts"][0]["text"] == SYSTEM_PROMPT
    assert "Khi nào sinh viên bị cảnh báo học tập?" in payload["contents"][0]["parts"][0]["text"]
    assert "[Tài liệu 1]" in payload["contents"][0]["parts"][0]["text"]


def test_generate_answer_gemini_with_history() -> None:
    gemini_response = BytesIO(
        json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "- Trả lời tiếp theo."}]
                        }
                    }
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    history = [
        {"role": "user", "content": "Câu hỏi trước"},
        {"role": "assistant", "content": "Trả lời trước"},
    ]

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("app.rag.from_scratch.llm.urlopen", return_value=gemini_response) as mocked_urlopen:
            answer = generate_answer("Câu hỏi mới", "Ngữ cảnh", history=history)

    assert answer == "- Trả lời tiếp theo."
    payload = json.loads(mocked_urlopen.call_args.args[0].data.decode("utf-8"))
    assert len(payload["contents"]) == 3
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "Câu hỏi trước"
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["text"] == "Trả lời trước"
    assert payload["contents"][2]["role"] == "user"


def test_generate_answer_gemini_missing_api_key_raises_value_error() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        with patch("app.rag.from_scratch.llm.GEMINI_API_KEY", ""):
            with pytest.raises(ValueError, match="Chưa cấu hình GEMINI_API_KEY"):
                generate_answer("Câu hỏi", "Ngữ cảnh")


def test_generate_answer_gemini_retries_transient_http_error() -> None:
    err_503 = HTTPError("url", 503, "Service Unavailable", None, None)
    success_resp = BytesIO(
        json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "- Đã thử lại thành công."}]}}
                ]
            }
        ).encode("utf-8")
    )

    mocked_urlopen = MagicMock(side_effect=[err_503, success_resp])
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("app.rag.from_scratch.llm.urlopen", mocked_urlopen):
            with patch("time.sleep") as mocked_sleep:
                answer = generate_answer("Câu hỏi", "Ngữ cảnh")

    assert answer == "- Đã thử lại thành công."
    assert mocked_urlopen.call_count == 2
    mocked_sleep.assert_called_once()


def test_system_prompt_rules() -> None:
    assert "TRẢ LỜI NGẮN GỌN" in SYSTEM_PROMPT
    assert "ĐIỀU KHOẢN TRÍCH DẪN" in SYSTEM_PROMPT
    assert "Chỉ sử dụng thông tin có trong [Tài liệu tham khảo]" in SYSTEM_PROMPT
    assert "Không tìm thấy thông tin liên quan" in SYSTEM_PROMPT
