from unittest.mock import patch

from fastapi import HTTPException

from app.main import app
from app.routes.chat import chat_endpoint, get_rag_index
from app.schemas.chat import ChatRequest


def test_openapi_contains_chat_endpoint() -> None:
    operation = app.openapi()["paths"]["/api/v1/chat"]["post"]

    assert operation["tags"] == ["Chat"]
    assert operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ChatResponse"}


def test_chat_endpoint_uses_cached_index() -> None:
    index = ([{"text": "Nội dung"}], object(), [[1.0]])
    get_rag_index.cache_clear()

    with (
        patch(
            "app.routes.chat.load_index",
            return_value=index,
        ) as mocked_load_index,
        patch(
            "app.routes.chat.rag_chat",
            return_value="Câu trả lời.",
        ) as mocked_rag_chat,
    ):
        first_response = chat_endpoint(ChatRequest(question="Câu hỏi 1?"))
        second_response = chat_endpoint(ChatRequest(question="Câu hỏi 2?"))

    get_rag_index.cache_clear()

    assert first_response.answer == "Câu trả lời."
    assert second_response.answer == "Câu trả lời."
    mocked_load_index.assert_called_once_with()
    assert mocked_rag_chat.call_count == 2
    assert mocked_rag_chat.call_args.args[0] == "Câu hỏi 2?"


def test_chat_endpoint_hides_internal_errors() -> None:
    get_rag_index.cache_clear()

    with patch(
        "app.routes.chat.load_index",
        side_effect=FileNotFoundError,
    ):
        try:
            chat_endpoint(ChatRequest(question="Câu hỏi?"))
        except HTTPException as error:
            assert error.status_code == 500
            assert error.detail == "Không thể xử lý câu hỏi."
        else:
            raise AssertionError("Expected HTTPException to be raised")
        finally:
            get_rag_index.cache_clear()


if __name__ == "__main__":
    test_openapi_contains_chat_endpoint()
    test_chat_endpoint_uses_cached_index()
    test_chat_endpoint_hides_internal_errors()
    print("All tests passed!")
