"""
Test Cross Encoder Reranker
"""

from backend.retrieval.cross_encoder import CrossEncoderReranker


def main():

    query = "How do Docker Personal Access Tokens work?"

    contexts = [

        {
            "type": "vector",
            "content": (
                "Docker Hub supports Personal Access Tokens "
                "for secure authentication."
            )
        },

        {
            "type": "vector",
            "content": (
                "Docker Desktop can be installed on "
                "Windows, macOS, and Linux."
            )
        },

        {
            "type": "graph",
            "content": (
                "Docker USES Personal Access Tokens"
            )
        },

        {
            "type": "graph",
            "content": (
                "Docker Desktop RUNS_ON Windows"
            )
        },

        {
            "type": "vector",
            "content": (
                "Containerd is the container runtime "
                "used by Docker."
            )
        }

    ]

    reranker = CrossEncoderReranker()

    ranked = reranker.rerank(
        query=query,
        contexts=contexts,
        top_k=5
    )

    print("=" * 80)
    print("RERANKED RESULTS\n")

    for index, context in enumerate(ranked, start=1):

        print(f"{index}. Score : {context['score']:.4f}")
        print(f"   Type  : {context['type']}")
        print(f"   Text  : {context['content']}")
        print()


if __name__ == "__main__":
    main()