from backend.indexing.loader import (
    find_chunk_files,
    load_chunk,
)

from backend.embeddings.encoder import (
    EmbeddingEncoder,
)

from backend.vector_db.uploader import (
    upload_embeddings,
)


def process_indexing(batch_size: int = 64):
    """
    Production indexing pipeline.

    Workflow:
    1. Find all chunk JSON files.
    2. Load chunks in batches.
    3. Generate embeddings in batches.
    4. Upload embeddings to Qdrant.
    """

    print("\n" + "=" * 80)
    print("LEXMIND AI DOCUMENT INDEXER")
    print("=" * 80)

    encoder = EmbeddingEncoder()

    chunk_files = find_chunk_files()

    total = len(chunk_files)

    print(f"\nFound {total} chunk files.\n")

    uploaded = 0

    for start in range(0, total, batch_size):

        batch_files = chunk_files[start:start + batch_size]

        # Load chunks
        chunks = [
            load_chunk(file)
            for file in batch_files
        ]

        # Extract text
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = encoder.encode_batch(texts)

        # Upload batch to Qdrant
        uploaded += upload_embeddings(
            chunks,
            embeddings
        )

        print(
            f"[{uploaded}/{total}] Indexed"
        )

    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    print(f"Successfully indexed {uploaded} chunks.")