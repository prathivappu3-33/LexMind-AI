"""
Chat Service

Provides a service layer between the API routes
and the RAG pipeline.
"""

from backend.pipeline.rag_pipeline import RAGPipeline


class ChatService:
    """Service responsible for handling chat requests."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ):
        """
        Generate an answer using the RAG pipeline.
        """

        return self.pipeline.answer(
            query=query,
            session_id=session_id,
            top_k=top_k,
        )