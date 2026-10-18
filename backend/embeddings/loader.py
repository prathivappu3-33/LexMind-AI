from pathlib import Path
import json


CHUNK_DIR = Path("datasets/chunks")


def find_chunk_files():
    """
    Find every chunk JSON file.
    """

    return sorted(CHUNK_DIR.rglob("*.json"))


def load_chunk(file_path: Path):
    """
    Load a single chunk JSON.

    Args:
        file_path: Path to chunk JSON.

    Returns:
        Chunk dictionary.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)