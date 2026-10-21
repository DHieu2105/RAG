from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse, ChatSource


def assert_raises_validation_error(function) -> None:
    try:
        function()
    except ValidationError:
        return

    raise AssertionError("Expected ValidationError to be raised")


def test_chat_request_accepts_a_question() -> None:
    request = ChatRequest(
        question="Sinh viên bị cảnh báo học tập khi nào?"
    )

    assert request.question == "Sinh viên bị cảnh báo học tập khi nào?"


def test_chat_request_rejects_an_empty_question() -> None:
    assert_raises_validation_error(lambda: ChatRequest(question=""))


def test_chat_response_contains_an_answer() -> None:
    response = ChatResponse(answer="Sinh viên được cảnh báo theo Điều 11.")

    assert response.answer == "Sinh viên được cảnh báo theo Điều 11."


def test_chat_source_can_expose_claim_level_grounding() -> None:
    source = ChatSource(
        article=11,
        clauses=[2],
        claims=["Ba lần cảnh báo liên tiếp thì bị buộc thôi học."],
        evidence=["Thuộc diện cảnh báo kết quả học tập 3 lần liên tiếp."],
    )

    assert source.clauses == [2]
    assert source.claims == [
        "Ba lần cảnh báo liên tiếp thì bị buộc thôi học."
    ]
    assert source.evidence == [
        "Thuộc diện cảnh báo kết quả học tập 3 lần liên tiếp."
    ]


if __name__ == "__main__":
    test_chat_request_accepts_a_question()
    test_chat_request_rejects_an_empty_question()
    test_chat_response_contains_an_answer()
    print("All tests passed!")
