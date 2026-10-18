"""
Custom exceptions used throughout LexMindAI.
"""

from fastapi import status


class LexMindAIException(Exception):
    """Base exception for LexMindAI."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class Neo4jConnectionException(LexMindAIException):
    """Raised when Neo4j is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class QdrantConnectionException(LexMindAIException):
    """Raised when Qdrant is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class OllamaConnectionException(LexMindAIException):
    """Raised when Ollama is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class RetrievalException(LexMindAIException):
    """Raised when retrieval fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class GenerationException(LexMindAIException):
    """Raised when answer generation fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR