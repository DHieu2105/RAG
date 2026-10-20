"""Schemas for the chat API."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A previous message in the conversation."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """A student's chat question."""

    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatSource(BaseModel):
    """A source reference returned by the retriever."""

    part: str | None = None
    chapter: str | None = None
    article: int | None = None
    part_title: str | None = None
    chapter_title: str | None = None
    article_title: str | None = None
    clauses: list[int] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class ChatResponse(BaseModel):
    """The answer returned by the RAG pipeline."""

    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
