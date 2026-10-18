"""
Memory Manager

Coordinates short-term conversation memory
and long-term semantic memory ingestion.
"""

from backend.memory.conversation import ConversationStore
from backend.memory.memory_ingestion import MemoryIngestionPipeline


class MemoryManager:

    def __init__(self):

        self.store = ConversationStore()

        self.ingestion = MemoryIngestionPipeline()

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ):

        self.store.add_message(
            session_id=session_id,
            role="user",
            content=message,
        )

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ):

        self.store.add_message(
            session_id=session_id,
            role="assistant",
            content=message,
        )

        #
        # Trigger long-term memory update
        #
        conversation = self.store.get_messages(session_id)

        self.ingestion.ingest(conversation)

    def get_recent_messages(
        self,
        session_id: str,
    ):

        return self.store.get_messages(session_id)

    def clear_memory(
        self,
        session_id: str,
    ):

        self.store.clear_session(session_id)

    def has_session(
        self,
        session_id: str,
    ):

        return self.store.has_session(session_id)

    def active_sessions(self):

        return self.store.active_sessions()

    def close(self):

        self.ingestion.close()