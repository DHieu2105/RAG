import json
from io import BytesIO
from unittest.mock import patch

from app.rag.from_scratch.llm import (
    GROUNDED_ANSWER_SCHEMA,
    INSUFFICIENT_ANSWER,
    MODEL_NAME,
    OLLAMA_URL,
    _concise_reader_claim,
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


def test_generate_grounded_answer_formats_multiple_claims_as_bullets() -> None:
    claims = [
        {
            "text": "Miễn 100%: sinh viên thuộc nhóm Alpha.",
            "citations": [
                {
                    "document": 1,
                    "clauses": [1],
                    "evidence": "Nhóm Alpha được miễn 100% học phí.",
                }
            ],
        },
        {
            "text": "Giảm 70%: sinh viên thuộc nhóm Beta.",
            "citations": [
                {
                    "document": 2,
                    "clauses": [2],
                    "evidence": "Nhóm Beta được giảm 70% học phí.",
                }
            ],
        },
    ]
    ollama_response = BytesIO(
        json.dumps(
            {"message": {"content": json.dumps({"claims": claims})}}
        ).encode("utf-8")
    )

    with patch(
        "app.rag.from_scratch.llm.urlopen",
        return_value=ollama_response,
    ):
        result = generate_grounded_answer("Câu hỏi", "Tài liệu")

    assert result["answer"] == (
        "- Miễn 100%: sinh viên thuộc nhóm Alpha.\n"
        "- Giảm 70%: sinh viên thuộc nhóm Beta."
    )
    assert len(result["citations"]) == 2


def test_generate_grounded_answer_summarizes_examples_for_broad_question() -> None:
    long_claim = (
        "Miễn 100% học phí: Đối tượng theo Pháp lệnh Ưu đãi người có công "
        "với cách mạng (anh hùng lực lượng vũ trang, thương binh, bệnh binh, "
        "thân nhân liệt sĩ và nhiều trường hợp con khác được liệt kê trong "
        "quy định áp dụng cho sinh viên đang theo học tại trường)."
    )
    evidence = (
        "a. Các đối tượng theo Pháp lệnh Ưu đãi người có công với cách mạng. "
        "Cụ thể: Anh hùng lực lượng vũ trang; thương binh; bệnh binh."
    )
    ollama_response = BytesIO(
        json.dumps(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "claims": [
                                {
                                    "text": long_claim,
                                    "citations": [
                                        {
                                            "document": 1,
                                            "clauses": [1],
                                            "evidence": evidence,
                                        }
                                    ],
                                }
                            ]
                        }
                    )
                }
            }
        ).encode("utf-8")
    )

    with patch(
        "app.rag.from_scratch.llm.urlopen",
        return_value=ollama_response,
    ):
        result = generate_grounded_answer(
            "Ai được miễn, giảm học phí?",
            "Tài liệu",
        )

    expected = (
        "Miễn 100% học phí: Đối tượng theo Pháp lệnh Ưu đãi người có công "
        "với cách mạng."
    )
    assert result["answer"] == expected
    assert result["citations"][0]["claim"] == expected
    assert result["citations"][0]["evidence"] == evidence


def test_concise_reader_claim_uses_evidence_when_claim_has_no_label_colon() -> None:
    claim = (
        "Sinh viên được miễn 100% học phí nếu thuộc diện người có công, bao gồm "
        + " ".join(["nhiều trường hợp"] * 20)
    )
    evidence = (
        "1. Đối tượng sinh viên được miễn 100% học phí "
        "a. Các đối tượng theo Pháp lệnh Ưu đãi người có công với cách mạng. "
        "Cụ thể: Anh hùng; thương binh; bệnh binh."
    )

    concise = _concise_reader_claim(
        "Ai được miễn học phí?",
        claim,
        [{"evidence": evidence}],
    )

    assert concise == (
        "Đối tượng sinh viên được miễn 100% học phí: Các đối tượng theo "
        "Pháp lệnh Ưu đãi người có công với cách mạng."
    )


if __name__ == "__main__":
    test_generate_answer()
    print("All tests passed!")
