"""
Gemma Client

Handles communication with the local Gemma model
running through Ollama.
"""

from ollama import Client


DEFAULT_MODEL = "gemma3:1b"


class GemmaClient:
    """
    Client for interacting with a locally running Gemma model.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an answer from Gemma.
        """

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
        )

        return response["response"].strip()