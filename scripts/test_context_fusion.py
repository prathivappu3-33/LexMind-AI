from backend.retrieval.vector_retriever import VectorRetriever
from backend.retrieval.graph_retriever import GraphRetriever
from backend.retrieval.context_fusion import ContextFusion

query = "How does Docker Hub implement multi-factor authentication?"

vector = VectorRetriever()

graph = GraphRetriever()

fusion = ContextFusion()

vector_results = vector.retrieve(query)

graph_results = graph.retrieve(query)

fused = fusion.fuse(
    vector_results,
    graph_results
)

print("=" * 100)
print("FUSED CONTEXT\n")

for item in fused:

    print(item)

print("=" * 100)