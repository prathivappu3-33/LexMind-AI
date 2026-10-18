"""
Prompt Builder

Builds a structured prompt for the LLM using
conversation memory, retrieved contexts,
and the user's query.
"""

from typing import Dict, List

from backend.memory.memory_models import Message


class PromptBuilder:
    """
    Constructs prompts for the LLM.
    """

    def build(
        self,
        query: str,
        contexts: List[Dict],
        memory: List[Message] | None = None,
    ) -> str:
        """
        Build the complete prompt.
        """

        sections = [
            self._build_system_prompt(),
            self._build_memory(memory),
            self._build_context(contexts),
            self._build_question(query),
            self._build_answer_header(),
        ]

        return "\n\n".join(
            section
            for section in sections
            if section
        )

    def _build_system_prompt(self) -> str:
        return """You are LexMindAI.

You are an AI assistant specialized in answering questions using ONLY the provided documentation and the conversation memory.

Instructions:

- Use conversation memory to understand follow-up questions.
- Answer ONLY using the supplied context.
- Never fabricate information.
- If the answer is unavailable, clearly state it.
- Do not use outside knowledge.
- Keep answers concise and technically accurate.
- Use Markdown formatting.
"""

    def _build_memory(
        self,
        memory: List[Message] | None,
    ) -> str:

        if not memory:
            return ""

        lines = [
            "==================== CONVERSATION MEMORY ====================\n"
        ]

        for message in memory:
            lines.append(
                f"{message.role.upper()}: {message.content}"
            )

        return "\n".join(lines)

    def _build_context(
        self,
        contexts: List[Dict],
    ) -> str:

        lines = [
            "==================== RETRIEVED CONTEXT ====================\n"
        ]

        if not contexts:
            lines.append("No context available.")
            return "\n".join(lines)

        for index, context in enumerate(contexts, start=1):

            lines.append(f"Context {index}")
            lines.append("-" * 50)
            lines.append(context["content"])
            lines.append("")

        return "\n".join(lines)

    def _build_question(
        self,
        query: str,
    ) -> str:

        return (
            "==================== QUESTION ====================\n\n"
            f"{query}"
        )

    def _build_answer_header(self) -> str:

        return "==================== ANSWER ====================\n"