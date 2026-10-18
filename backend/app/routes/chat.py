"""HTTP endpoint for the RAG chat pipeline."""

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import chat_with_sources
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter()
rag_chat = chat_with_sources


@lru_cache(maxsize=1)
def get_rag_index():
    """Load the persisted index once for the lifetime of this process."""
    return load_index()


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Answer a student's question using the cached RAG index."""
    try:
        chunks, vectorizer, document_vectors = get_rag_index()
        result = rag_chat(
            request.question,
            chunks,
            vectorizer,
            document_vectors,
            [message.model_dump() for message in request.history],
        )
        if isinstance(result, str):
            return ChatResponse(answer=result)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý câu hỏi.",
        ) from exc
