"""
Answer Generator

Combines the PromptBuilder and GemmaClient
to generate answers from retrieved contexts
and conversation memory.
"""

from typing import Dict, List

from backend.llm.gemma_client import GemmaClient
from backend.llm.prompt_builder import PromptBuilder
from backend.memory.memory_models import Message


class AnswerGenerator:
    """
    Generates answers using PromptBuilder
    and GemmaClient.
    """

    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.gemma_client = GemmaClient()

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        memory: List[Message] | None = None,
    ) -> str:
        """
        Generate an answer using
        retrieved context and conversation memory.
        """

        prompt = self.prompt_builder.build(
            query=query,
            contexts=contexts,
            memory=memory,
        )

        answer = self.gemma_client.generate(
            prompt=prompt,
        )

        return answer