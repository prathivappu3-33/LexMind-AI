"""
API Routes for LexMindAI.
"""

from fastapi import APIRouter

from backend.api.dependencies import chat_service
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
)

router = APIRouter()


@router.get("/")
async def root():
    return {
        "application": "LexMindAI",
        "version": "0.1.0",
        "status": "running",
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):

    result = chat_service.chat(
        query=request.query,
        session_id=request.session_id,
        top_k=request.top_k,
    )

    citations = []

    for context in result["contexts"]:

        if context.get("type") != "vector":
            continue

        metadata = context.get("metadata", {})

        citations.append(
            Citation(
                document=metadata.get("source", "Unknown"),
                chunk_id=metadata.get("chunk_id", "Unknown"),
                score=round(float(context.get("score", 0.0)), 4),
            )
        )

    return ChatResponse(
        session_id=result["session_id"],
        answer=result["answer"],
        confidence=result["confidence"],
        citations=citations,
    )