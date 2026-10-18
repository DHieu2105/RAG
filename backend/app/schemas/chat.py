"""Schemas for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A student's chat question."""

    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """The answer returned by the RAG pipeline."""

    answer: str
