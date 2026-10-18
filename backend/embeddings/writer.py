from pathlib import Path
import numpy as np
import json


EMBEDDING_DIR = Path("datasets/embeddings")


def save_embedding(result: dict):

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chunk_id = result["chunk_id"]

    vector = result["embedding"]

    metadata = {

        "chunk_id": result["chunk_id"],

        "document_id": result["document_id"],

        "source": result["source"],

        "title": result["title"],

        "section_heading": result["section_heading"]

    }

    np.save(
        EMBEDDING_DIR / f"{chunk_id}.npy",
        vector
    )

    with open(
        EMBEDDING_DIR / f"{chunk_id}.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )