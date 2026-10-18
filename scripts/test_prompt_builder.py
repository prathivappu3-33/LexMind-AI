from backend.llm.prompt_builder import PromptBuilder


def main():

    builder = PromptBuilder()

    query = "How do I recover my Docker account?"

    contexts = [
        {
            "type": "vector",
            "content": (
                "If you lose your recovery code, sign in to Docker, "
                "go to Account Settings, open the 2FA section, and "
                "generate a new recovery code."
            ),
        },
        {
            "type": "vector",
            "content": (
                "If you lose both your authentication device and "
                "recovery code, contact Docker Support."
            ),
        },
        {
            "type": "graph",
            "content": "Docker HAS_PART Docker Desktop",
        },
    ]

    prompt = builder.build(
        query=query,
        contexts=contexts,
    )

    print("\n")
    print("=" * 100)
    print(prompt)
    print("=" * 100)


if __name__ == "__main__":
    main()