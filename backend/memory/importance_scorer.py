"""
Importance Scorer

Determines whether a conversation summary
should be stored in long-term memory.
"""

from backend.llm.gemma_client import GemmaClient


class ImportanceScorer:
    """
    Scores conversation summaries.

    Returns:
        score (0-100)
        should_store (True/False)
    """

    def __init__(self):
        self.llm = GemmaClient()

    def score(
        self,
        summary: str,
    ) -> tuple[int, bool]:
        """
        Score the importance of a summary.
        """

        if not summary.strip():
            return 0, False

        prompt = f"""
You are an AI memory evaluation system.

Evaluate whether the following conversation summary
is important enough to store in long-term memory.

Score from 0 to 100.

Guidelines:

0-20:
Greetings, thanks, casual conversation.

21-40:
Simple questions with little long-term value.

41-60:
Useful factual discussion.

61-80:
Technical discussion,
problem solving,
or significant decisions.

81-100:
Critical knowledge,
important discoveries,
or information likely to be useful in future conversations.

Return ONLY the score.

Summary:

{summary}
"""

        response = self.llm.generate(
            prompt=prompt,
        ).strip()

        try:
            score = int(response)
        except ValueError:
            score = 50

        score = max(0, min(score, 100))

        should_store = score >= 60

        return score, should_store