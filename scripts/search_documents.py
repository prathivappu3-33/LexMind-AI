from backend.retrieval.retriever import Retriever


def main():

    retriever = Retriever()

    query = "What hash identifies a Git commit according to the Docker documentation?"

    results = retriever.search(
        query=query,
        top_k=5
    )

    print("\n")

    for index, result in enumerate(results, start=1):

        print("=" * 80)

        print(f"Result {index}")

        print(f"Score : {result.score:.4f}")

        print()

        print(result.payload["title"])

        print()

        print(result.payload["text"][:300])

        print()

        print(result.payload["source"])


if __name__ == "__main__":
    main()