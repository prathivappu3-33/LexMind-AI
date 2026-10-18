"""
LexMindAI FastAPI Application

This module initializes the FastAPI application,
registers API routes, configures logging,
and registers global exception handlers.
"""

from fastapi import FastAPI

from backend.api.routes import router
from backend.config import setup_logging
from backend.exceptions import register_exception_handlers

# ---------------------------------------------------------
# Initialize logging
# ---------------------------------------------------------

setup_logging()

# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="LexMindAI API",
    description="Hybrid Graph RAG API for Intelligent Knowledge Retrieval",
    version="0.1.0",
)

# ---------------------------------------------------------
# Register global exception handlers
# ---------------------------------------------------------

register_exception_handlers(app)

# ---------------------------------------------------------
# Register API routes
# ---------------------------------------------------------

app.include_router(router)