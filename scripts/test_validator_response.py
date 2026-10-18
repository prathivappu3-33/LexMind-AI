from backend.llm.gemma_client import GemmaClient


def main():
    llm = GemmaClient()

    prompt = """
You are validating retrieval results.

Question:
What is Docker?

Candidate Chunk:
Docker is an open platform for developing, shipping, and running applications.
Docker allows you to separate your applications from your infrastructure
so you can deliver software quickly.

Return ONLY valid JSON:
{
  "relevant": true,
  "confidence": 0.95,
  "reason": "short reason"
}
"""

    print("\nSending prompt to Gemma...\n")

    response = llm.generate(prompt)

    print("=" * 80)
    print("RAW GEMMA RESPONSE")
    print("=" * 80)

    print(repr(response))

    print("\n" + "=" * 80)
    print("NORMAL RESPONSE")
    print("=" * 80)

    print(response)

    print("=" * 80)


if __name__ == "__main__":
    main()