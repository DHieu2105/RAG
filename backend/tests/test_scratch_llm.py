import json
from io import BytesIO
from unittest.mock import patch

from app.rag.from_scratch.llm import (
    MODEL_NAME,
    OLLAMA_URL,
    generate_answer,
)


def test_generate_answer() -> None:
    ollama_response = BytesIO(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": "Sinh viên bị cảnh báo theo Điều 11, trang 23.",
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    with patch(
        "app.rag.from_scratch.llm.urlopen",
        return_value=ollama_response,
    ) as mocked_urlopen:
        answer = generate_answer(
            "Khi nào sinh viên bị cảnh báo học tập?",
            "[Tài liệu 1]\nĐiều: 11\nTrang PDF: 23",
        )

    assert answer == "Sinh viên bị cảnh báo theo Điều 11, trang 23."
    mocked_urlopen.assert_called_once()

    request = mocked_urlopen.call_args.args[0]
    assert request.full_url == OLLAMA_URL
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"

    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == MODEL_NAME
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "Chỉ trả lời dựa trên tài liệu" in (
        payload["messages"][0]["content"]
    )
    assert payload["messages"][1]["role"] == "user"
    assert "Khi nào sinh viên bị cảnh báo học tập?" in (
        payload["messages"][1]["content"]
    )
    assert "[Tài liệu 1]" in payload["messages"][1]["content"]


if __name__ == "__main__":
    test_generate_answer()
    print("All tests passed!")
