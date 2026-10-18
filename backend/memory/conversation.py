"""
Short-Term Conversation Store.

Maintains active conversations in memory using
a sliding window for recent messages.
"""

from collections import defaultdict
from typing import Dict, List

from backend.memory.memory_models import (
    Conversation,
    Message,
)


class ConversationStore:
    """
    Stores active conversations in RAM.

    One Conversation object is maintained
    for each session_id.
    """

    def __init__(
        self,
        max_messages: int = 20,
    ):
        self.max_messages = max_messages
        self._sessions: Dict[str, Conversation] = {}

    def create_session(
        self,
        session_id: str,
    ) -> None:
        """
        Create a new session if it doesn't exist.
        """

        if session_id not in self._sessions:
            self._sessions[session_id] = Conversation(
                session_id=session_id
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to the session.
        """

        self.create_session(session_id)

        conversation = self._sessions[session_id]

        conversation.messages.append(
            Message(
                role=role,
                content=content,
            )
        )

        # Sliding window
        if len(conversation.messages) > self.max_messages:
            conversation.messages = conversation.messages[
                -self.max_messages :
            ]

    def get_messages(
        self,
        session_id: str,
    ) -> List[Message]:
        """
        Return all messages in the session.
        """

        if session_id not in self._sessions:
            return []

        return self._sessions[session_id].messages

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Remove a session from memory.
        """

        self._sessions.pop(
            session_id,
            None,
        )

    def has_session(
        self,
        session_id: str,
    ) -> bool:
        return session_id in self._sessions

    def active_sessions(self) -> int:
        """
        Number of active conversations.
        """

        return len(self._sessions)