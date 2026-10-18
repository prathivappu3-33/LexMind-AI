from backend.pipeline.retrieval_pipeline import RetrievalPipeline


def main():

    pipeline = RetrievalPipeline()

    query = "How do I recover my Docker account?"

    results = pipeline.retrieve(
        query=query,
        top_k=5,
    )

    print("\n" + "=" * 80)

    for idx, result in enumerate(results, start=1):

        print(f"\nRank : {idx}")

        print(f"Type : {result['type']}")

        print(f"Score: {result['score']:.4f}")

        print()

        print(result["content"])

        print("=" * 80)


if __name__ == "__main__":
    main()