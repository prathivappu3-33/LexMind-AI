from backend.embeddings.encoder import EmbeddingEncoder


class EmbeddingProcessor:

    def __init__(self):

        self.encoder = EmbeddingEncoder()

    def process_chunk(self, chunk: dict):

        embedding = self.encoder.encode(
            chunk["text"]
        )

        return {

            "chunk_id": chunk["chunk_id"],

            "document_id": chunk["document_id"],

            "source": chunk["source"],

            "title": chunk["title"],

            "section_heading": chunk["section_heading"],

            "embedding": embedding

        }