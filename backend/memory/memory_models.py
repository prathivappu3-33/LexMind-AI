"""
Memory models for LexMindAI.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    Single conversation message.
    """

    role: str = Field(
        description="user or assistant",
    )

    content: str

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
    )


class Conversation(BaseModel):
    """
    One complete conversation session.
    """

    session_id: str

    messages: List[Message] = Field(
        default_factory=list,
    )


class MemoryContext(BaseModel):
    """
    Memory injected into the prompt.
    """

    conversation: str