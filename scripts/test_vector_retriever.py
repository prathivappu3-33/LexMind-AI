from backend.retrieval.vector_retriever import VectorRetriever

retriever = VectorRetriever()

results = retriever.retrieve(
    "How does Docker Hub implement multi-factor authentication?"
)

print("=" * 100)
print("VECTOR RETRIEVER RESULTS\n")

for i, result in enumerate(results, start=1):

    print(f"Result {i}")

    print("Score:", round(result["score"], 4))
    print("Title:", result["title"])
    print("Section:", result["section_heading"])
    print("Source:", result["source"])

    print()
    print(result["text"][:300])
    print()

    print("-" * 100)