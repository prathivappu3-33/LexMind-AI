"""
Shared application dependencies.
"""

from backend.pipeline.rag_pipeline import RAGPipeline
from backend.services.chat_service import ChatService

# Single shared RAG pipeline instance
rag_pipeline = RAGPipeline()

# Single shared chat service
chat_service = ChatService(rag_pipeline)