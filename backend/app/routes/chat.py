"""HTTP endpoint for the RAG chat pipeline."""

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.rag.from_scratch.index_store import load_index
from app.rag.from_scratch.pipeline import chat as rag_chat
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter()


@lru_cache(maxsize=1)
def get_rag_index():
    """Load the persisted index once for the lifetime of this process."""
    return load_index()


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Answer a student's question using the cached RAG index."""
    try:
        chunks, vectorizer, document_vectors = get_rag_index()
        answer = rag_chat(
            request.question,
            chunks,
            vectorizer,
            document_vectors,
        )
        return ChatResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý câu hỏi.",
        ) from exc
