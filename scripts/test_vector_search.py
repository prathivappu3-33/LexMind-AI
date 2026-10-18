from backend.retrieval.query_processor import QueryProcessor
from backend.vector_db.search import search_embeddings

processor = QueryProcessor()

query = "How does Docker Hub implement multi-factor authentication?"

query_object = processor.process(query)

results = search_embeddings(
    query_object["embedding"],
    top_k=5
)

print("=" * 100)
print("QUERY")
print(query)

print("\nTOP RESULTS\n")

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