import json
from io import BytesIO
from unittest.mock import patch

from app.rag.from_scratch.llm import (
    GROUNDED_ANSWER_SCHEMA,
    INSUFFICIENT_ANSWER,
    MODEL_NAME,
    OLLAMA_URL,
    generate_answer,
    generate_grounded_answer,
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
    assert "tất cả đều là nguồn" in payload["messages"][0]["content"]
    assert "chưa đủ thông tin" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"
    assert "Khi nào sinh viên bị cảnh báo học tập?" in (
        payload["messages"][1]["content"]
    )
    assert "[Tài liệu 1]" in payload["messages"][1]["content"]
    assert payload["options"]["temperature"] == 0.1
    assert payload["think"] is False


def test_generate_grounded_answer_returns_explicit_citations() -> None:
    ollama_response = BytesIO(
        json.dumps(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "claims": [
                                {
                                    "text": "Câu trả lời có căn cứ.",
                                    "citations": [
                                        {
                                            "document": 1,
                                            "clauses": [2],
                                            "evidence": "Nội dung khoản 2.",
                                        }
                                    ],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    with patch(
        "app.rag.from_scratch.llm.urlopen",
        return_value=ollama_response,
    ) as mocked_urlopen:
        result = generate_grounded_answer(
            "Câu hỏi",
            "[Tài liệu 1]\nĐiều: 11\nKhoản: 1, 2",
            answer_requirements=["Nêu ngưỡng cảnh báo."],
        )

    assert result == {
        "answer": "Câu trả lời có căn cứ.",
        "citations": [
            {
                "claim": "Câu trả lời có căn cứ.",
                "document": 1,
                "clauses": [2],
                "evidence": "Nội dung khoản 2.",
            }
        ],
    }
    payload = json.loads(mocked_urlopen.call_args.args[0].data.decode("utf-8"))
    assert payload["format"] == GROUNDED_ANSWER_SCHEMA
    assert "maxItems" not in payload["format"]["properties"]["claims"]
    assert payload["options"]["temperature"] == 0
    assert payload["think"] is False
    assert "danh sách claim độc lập" in payload["messages"][0]["content"]
    assert "không giới hạn số claim" in payload["messages"][0]["content"]
    assert "chỉ dựa vào khoản 2 thì trả `[2]`" in (
        payload["messages"][0]["content"]
    )
    assert "Các ý bắt buộc phải bao phủ" in payload["messages"][1]["content"]
    assert "Nêu ngưỡng cảnh báo." in payload["messages"][1]["content"]


def test_generate_grounded_answer_accepts_exact_insufficient_response() -> None:
    ollama_response = BytesIO(
        json.dumps(
            {
                "message": {
                    "content": json.dumps(
                        {"claims": []},
                        ensure_ascii=False,
                    )
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )

    with patch(
        "app.rag.from_scratch.llm.urlopen",
        return_value=ollama_response,
    ):
        result = generate_grounded_answer("Câu hỏi", "Tài liệu")

    assert result == {"answer": INSUFFICIENT_ANSWER, "citations": []}


if __name__ == "__main__":
    test_generate_answer()
    print("All tests passed!")
