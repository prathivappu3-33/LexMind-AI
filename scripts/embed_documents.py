from backend.embeddings.loader import (
    find_chunk_files,
    load_chunk,
)

from backend.embeddings.processor import EmbeddingProcessor

from backend.vector_db.uploader import upload_embedding


def main():

    files = find_chunk_files()

    chunk = load_chunk(files[0])

    processor = EmbeddingProcessor()

    result = processor.process_chunk(chunk)

    upload_embedding(result)

    print("Done!")


if __name__ == "__main__":
    main()