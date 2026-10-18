from backend.chunking.chunker import generate_chunks
from backend.chunking.writer import save_chunks


def process_chunks(document):
    """
    Generate and save chunks for one processed document.
    """     

    chunks = generate_chunks(document)

    save_chunks(chunks)

    return chunks