"""
Global exception handlers for LexMindAI.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.config import get_logger
from backend.exceptions.custom_exceptions import LexMindAIException

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI):
    """
    Register all global exception handlers.
    """

    @app.exception_handler(LexMindAIException)
    async def lexmind_exception_handler(
        request: Request,
        exc: LexMindAIException,
    ):
        """
        Handle all custom LexMindAI exceptions.
        """
        logger.exception(exc.message)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ):
        """
        Handle unexpected exceptions.
        """
        logger.exception("Unhandled exception")

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred.",
            },
        )