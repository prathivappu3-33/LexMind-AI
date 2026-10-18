from pathlib import Path
import json


CHUNK_DIRECTORY = Path("datasets/chunks")


def find_chunk_files():
    """
    Find every chunk JSON file.
    """

    return sorted(CHUNK_DIRECTORY.rglob("*.json"))


def load_chunk(file_path: Path) -> dict:
    """
    Load one chunk JSON.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)