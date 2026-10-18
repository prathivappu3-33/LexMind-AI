"""
RAG Pipeline

Coordinates retrieval, memory,
answer generation and confidence.
"""

from typing import Any
from uuid import uuid4

from backend.confidence import ConfidenceEngine
from backend.llm.answer_generator import AnswerGenerator
from backend.memory import MemoryManager
from backend.pipeline.retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    """
    End-to-end conversational Graph-RAG pipeline.
    """

    def __init__(self):

        self.retrieval_pipeline = RetrievalPipeline()

        self.answer_generator = AnswerGenerator()

        self.confidence_engine = ConfidenceEngine()

        self.memory_manager = MemoryManager()

    def answer(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Generate an answer with conversational memory.
        """

        #
        # Create session automatically
        #
        if session_id is None:
            session_id = str(uuid4())

        #
        # Retrieve memory
        #
        memory = self.memory_manager.get_recent_messages(
            session_id
        )

        #
        # Retrieve documentation
        #
        contexts = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
        )

        #
        # Generate answer
        #
        answer = self.answer_generator.generate(
            query=query,
            contexts=contexts,
            memory=memory,
        )

        #
        # Store conversation
        #
        self.memory_manager.add_user_message(
            session_id=session_id,
            message=query,
        )

        self.memory_manager.add_assistant_message(
            session_id=session_id,
            message=answer,
        )

        #
        # Confidence
        #
        confidence = self.confidence_engine.calculate(
            contexts
        )

        return {
            "session_id": session_id,
            "answer": answer,
            "contexts": contexts,
            "confidence": confidence,
        }