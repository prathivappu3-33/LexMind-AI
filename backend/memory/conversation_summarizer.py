"""
Conversation Summarizer

Converts long conversations into concise semantic
summaries before storing them in long-term memory.
"""

from typing import List

from backend.memory.memory_models import Message
from backend.llm.gemma_client import GemmaClient


class ConversationSummarizer:
    """
    Uses the LLM to summarize conversations.

    The summary should preserve:

    - Main discussion topics
    - Important entities
    - Decisions
    - Conclusions
    - Resolutions

    while removing conversational noise.
    """

    def __init__(self):
        self.llm = GemmaClient()

    def summarize(
        self,
        messages: List[Message],
    ) -> str:
        """
        Summarize a conversation.
        """

        if not messages:
            return ""

        conversation = "\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in messages
        )

        prompt = f"""
You are an expert AI memory system.

Your task is to summarize a conversation for long-term memory.

Rules:

1. Keep only important information.
2. Preserve technical details.
3. Preserve important decisions.
4. Preserve conclusions.
5. Preserve important entities.
6. Remove greetings.
7. Remove thank-you messages.
8. Remove repeated information.

Conversation:

{conversation}

Return ONLY the summary.
"""

        summary = self.llm.generate(
            prompt=prompt,
        )

        return summary.strip()