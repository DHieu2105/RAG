from unittest.mock import patch

from app.rag.from_scratch.pipeline import retrieve_for_chat
from app.routes.chat import chat_endpoint, get_bm25_index, get_rag_index
from app.schemas.chat import ChatRequest


def _clear_retrieval_caches() -> None:
    get_rag_index.cache_clear()
    get_bm25_index.cache_clear()


def test_retrieve_for_chat_uses_hybrid_by_default() -> None:
    chunks = [{"text": "Nội dung"}]
    vectorizer = object()
    bm25_index = object()
    expected = [{"text": "Hybrid"}]

    with (
        patch(
            "app.rag.from_scratch.pipeline.ask_hybrid",
            return_value=expected,
        ) as mocked_hybrid,
        patch(
            "app.rag.from_scratch.pipeline.ask_with_query_processing"
        ) as mocked_legacy,
    ):
        results = retrieve_for_chat(
            "Câu hỏi?",
            chunks,
            vectorizer,
            [[1.0]],
            bm25_index=bm25_index,
        )

    assert [result["text"] for result in results] == ["Hybrid"]
    assert results[0]["_context_complete"] is True
    mocked_hybrid.assert_called_once_with(
        "Câu hỏi?",
        chunks,
        vectorizer,
        [[1.0]],
        bm25_index,
        top_k=5,
    )
    mocked_legacy.assert_not_called()


def test_retrieve_for_chat_supports_legacy_rollback() -> None:
    chunks = [{"text": "Nội dung"}]
    expected = [{"text": "TF-IDF"}]

    with (
        patch(
            "app.rag.from_scratch.pipeline.ask_hybrid"
        ) as mocked_hybrid,
        patch(
            "app.rag.from_scratch.pipeline.ask_with_query_processing",
            return_value=expected,
        ) as mocked_legacy,
    ):
        results = retrieve_for_chat(
            "Câu hỏi?",
            chunks,
            object(),
            [[1.0]],
            retrieval_mode="tfidf_qp",
        )

    assert [result["text"] for result in results] == ["TF-IDF"]
    assert results[0]["_context_complete"] is True
    mocked_legacy.assert_called_once()
    mocked_hybrid.assert_not_called()


def test_chat_endpoint_caches_bm25_for_hybrid_requests() -> None:
    index = ([{"text": "Nội dung"}], object(), [[1.0]])
    bm25_index = object()
    _clear_retrieval_caches()

    with (
        patch.dict("app.routes.chat.os.environ", {}, clear=True),
        patch("app.routes.chat.load_index", return_value=index),
        patch(
            "app.routes.chat.build_bm25_index",
            return_value=bm25_index,
        ) as mocked_build_bm25,
        patch(
            "app.routes.chat.rag_chat",
            return_value="Câu trả lời.",
        ) as mocked_rag_chat,
    ):
        chat_endpoint(ChatRequest(question="Câu hỏi 1?"))
        chat_endpoint(ChatRequest(question="Câu hỏi 2?"))

    _clear_retrieval_caches()

    mocked_build_bm25.assert_called_once_with(index[0])
    assert mocked_rag_chat.call_count == 2
    assert mocked_rag_chat.call_args.kwargs["retrieval_mode"] == "hybrid"
    assert mocked_rag_chat.call_args.kwargs["bm25_index"] is bm25_index


def test_chat_endpoint_legacy_mode_does_not_build_bm25() -> None:
    index = ([{"text": "Nội dung"}], object(), [[1.0]])
    _clear_retrieval_caches()

    with (
        patch.dict(
            "app.routes.chat.os.environ",
            {"UNIGUIDE_RETRIEVAL_MODE": "tfidf_qp"},
            clear=True,
        ),
        patch("app.routes.chat.load_index", return_value=index),
        patch("app.routes.chat.build_bm25_index") as mocked_build_bm25,
        patch(
            "app.routes.chat.rag_chat",
            return_value="Câu trả lời.",
        ) as mocked_rag_chat,
    ):
        chat_endpoint(ChatRequest(question="Câu hỏi?"))

    _clear_retrieval_caches()

    mocked_build_bm25.assert_not_called()
    assert mocked_rag_chat.call_args.kwargs["retrieval_mode"] == "tfidf_qp"
    assert mocked_rag_chat.call_args.kwargs["bm25_index"] is None
