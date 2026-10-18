import json
from pathlib import Path


OUTPUT_DIR = Path("datasets/processed_documents")


def save_document(document: dict):
    """
    Save one processed document as JSON.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = Path(document["source"])

    filename = source.stem + ".json"

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            document,
            f,
            indent=4,
            ensure_ascii=False
        )

    return output_path