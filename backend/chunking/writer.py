import json
from pathlib import Path


CHUNK_DIR = Path("datasets/chunks")


def save_chunks(chunks: list):
    """
    Save all chunks from one document.

    Each document gets its own directory so chunks from
    different documents cannot overwrite each other.

    Example:

    datasets/chunks/context/context_000001.json
    datasets/chunks/context/context_000002.json
    """

    if len(chunks) == 0:
        return

    source = Path(chunks[0]["source"])

    # Use the document filename, not its parent directory.
    document_name = source.stem

    output_dir = CHUNK_DIR / document_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):

        filename = f"{document_name}_{i+1:06d}.json"

        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:

            json.dump(
                chunk,
                f,
                indent=4,
                ensure_ascii=False
            )
