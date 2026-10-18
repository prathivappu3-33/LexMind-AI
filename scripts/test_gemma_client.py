from backend.llm.gemma_client import GemmaClient


def main():

    client = GemmaClient()

    prompt = """
You are LexMindAI.

Question:
What is Docker?

Answer:
"""

    answer = client.generate(prompt)

    print("\n")
    print("=" * 80)
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    main()