from backend.retrieval.graph_retriever import GraphRetriever

retriever = GraphRetriever()

results = retriever.retrieve(
    "How does Docker use Personal Access Tokens?"
)


print("=" * 100)

print("GRAPH RETRIEVER RESULTS\n")

for result in results:

    print(result)

print()

print("=" * 100)