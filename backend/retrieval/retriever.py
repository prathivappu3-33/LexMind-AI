from backend.embeddings.encoder import EmbeddingEncoder
from backend.vector_db.client import get_client
from backend.vector_db.collection import COLLECTION_NAME


class Retriever:

    def __init__(self):
        self.encoder = EmbeddingEncoder()
        self.client = get_client()

    def search(self, query: str, top_k: int = 5):

        query_vector = self.encoder.encode(query)

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        return response.points