"""HTTP endpoint for the RAG chat pipeline."""

from functools import lru_cache
import os

from fastapi import APIRouter, HTTPException

from app.rag.from_scratch.bm25 import build_bm25_index
from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import (
    HYBRID_RETRIEVAL_MODE,
    SUPPORTED_RETRIEVAL_MODES,
    chat_with_sources,
)
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter()
rag_chat = chat_with_sources
RETRIEVAL_MODE_ENV = "UNIGUIDE_RETRIEVAL_MODE"


@lru_cache(maxsize=1)
def get_rag_index():
    """Load the persisted index once for the lifetime of this process."""
    return load_index()


@lru_cache(maxsize=1)
def get_bm25_index():
    """Build the BM25 index once and reuse it across chat requests."""
    chunks, _, _ = get_rag_index()
    return build_bm25_index(chunks)


def get_retrieval_mode() -> str:
    """Read the production retrieval mode with a legacy rollback option."""
    mode = os.getenv(RETRIEVAL_MODE_ENV, HYBRID_RETRIEVAL_MODE).strip().lower()
    if mode not in SUPPORTED_RETRIEVAL_MODES:
        supported = ", ".join(sorted(SUPPORTED_RETRIEVAL_MODES))
        raise ValueError(
            f"Invalid {RETRIEVAL_MODE_ENV}={mode!r}; expected {supported}."
        )
    return mode


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Answer a student's question using the cached RAG index."""
    try:
        chunks, vectorizer, document_vectors = get_rag_index()
        retrieval_mode = get_retrieval_mode()
        bm25_index = (
            get_bm25_index()
            if retrieval_mode == HYBRID_RETRIEVAL_MODE
            else None
        )
        result = rag_chat(
            request.question,
            chunks,
            vectorizer,
            document_vectors,
            history=[message.model_dump() for message in request.history],
            bm25_index=bm25_index,
            retrieval_mode=retrieval_mode,
        )
        if isinstance(result, str):
            return ChatResponse(answer=result)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý câu hỏi.",
        ) from exc
