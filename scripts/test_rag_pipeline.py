from backend.pipeline.rag_pipeline import RAGPipeline


def main():

    rag = RAGPipeline()

    query = input("Ask LexMindAI: ")

    print("\nSearching...\n")

    answer = rag.answer(query)

    print("=" * 100)
    print(answer)
    print("=" * 100)


if __name__ == "__main__":
    main()