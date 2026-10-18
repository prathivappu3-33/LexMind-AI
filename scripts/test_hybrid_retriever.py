"""
Test Hybrid Retriever
"""

from pprint import pprint

from backend.retrieval.hybrid_retriever import HybridRetriever


def main():

    retriever = HybridRetriever()

    query = input("\nEnter Query: ")

    results = retriever.retrieve(query)

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nRetrieved {len(results)} items\n")

    for i, result in enumerate(results, start=1):

        print("-" * 80)
        print(f"Result {i}")
        pprint(result)
        print("-" * 80)


if __name__ == "__main__":
    main()