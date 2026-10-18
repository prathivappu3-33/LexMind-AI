from typing import List
import uuid


DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 80


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation.

    English text averages around 0.75 words per token.
    """

    words = len(text.split())
    return int(words / 0.75)


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text: Input text.
        chunk_size: Approximate token size.
        overlap: Approximate token overlap.

    Returns:
        List of chunk strings.
    """

    words = text.split()

    approx_words = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)

    chunks = []

    start = 0

    while start < len(words):

        end = start + approx_words

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += approx_words - overlap_words

    return chunks


def generate_chunks(document: dict) -> List[dict]:
    """
    Generate chunks from one processed document.

    Args:
        document: Processed document dictionary.

    Returns:
        List of chunk dictionaries.
    """

    chunk_list = []

    for section_index, section in enumerate(document["sections"]):

        section_chunks = split_text(section["content"])

        for chunk_index, chunk_text in enumerate(section_chunks):

            chunk = {

                "chunk_id": str(uuid.uuid4()),

                "document_id": document["document_id"],

                "source": document["source"],

                "title": document["title"],

                "metadata": document["metadata"],

                "section_id": section_index,

                "section_heading": section["heading"],

                "section_level": section["level"],

                "chunk_index": chunk_index,

                "text": chunk_text,

                "token_count": estimate_tokens(chunk_text),

                "num_words": len(chunk_text.split()),

                "num_characters": len(chunk_text)

            }

            chunk_list.append(chunk)

    return chunk_list