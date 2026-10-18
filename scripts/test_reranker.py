from backend.reranker.reranker import Reranker


def main():

    reranker = Reranker()

    query = "How do I recover my Docker account?"

    contexts = [

        {
            "type": "vector",
            "content": "This page explains how to recover your Docker account."
        },

        {
            "type": "vector",
            "content": "Docker Desktop installation guide."
        },

        {
            "type": "vector",
            "content": "Inference engines for AI models."
        },

        {
            "type": "vector",
            "content": "Recover your Docker account using recovery codes."
        }

    ]

    ranked = reranker.rerank(
        query=query,
        contexts=contexts,
        top_k=4,
    )

    print()

    for i, context in enumerate(ranked, start=1):

        print("=" * 80)
        print(f"Rank : {i}")
        print(f"Score: {context['score']:.4f}")
        print()
        print(context["content"])


if __name__ == "__main__":
    main()