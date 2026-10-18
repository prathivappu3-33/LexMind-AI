from backend.llm.answer_generator import AnswerGenerator


def main():

    generator = AnswerGenerator()

    query = "How do I recover my Docker account?"

    contexts = [
        {
            "type": "vector",
            "content": (
                "Recovery codes can be regenerated "
                "from Account Settings."
            ),
        }
    ]

    answer = generator.generate(
        query=query,
        contexts=contexts,
    )

    print("\n")
    print("=" * 100)
    print(answer)
    print("=" * 100)


if __name__ == "__main__":
    main()