"""
Pydantic schemas for LexMindAI API.
"""

from pydantic import BaseModel, Field

from backend.confidence import ConfidenceResult


class Citation(BaseModel):
    """Single retrieved source."""

    document: str
    chunk_id: str
    score: float


class ChatRequest(BaseModel):
    """Incoming chat request."""

    session_id: str | None = Field(
        default=None,
        description="Conversation session identifier. Leave empty to start a new conversation.",
    )

    query: str = Field(
        ...,
        min_length=1,
        description="User question",
        examples=[
            "How do I enable two-factor authentication?"
        ],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieved contexts",
    )


class ChatResponse(BaseModel):
    """Chat response."""

    session_id: str

    answer: str

    confidence: ConfidenceResult

    citations: list[Citation] = Field(
        default_factory=list
    )