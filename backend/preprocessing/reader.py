from pathlib import Path


RAW_DATASET_DIR = Path("datasets/raw_documents")


def find_markdown_files():
    """
    Find only documentation markdown files.
    """

    markdown_files = []

    for file in RAW_DATASET_DIR.rglob("*.md"):

        # Only index documentation
        if "content" not in file.parts:
            continue

        markdown_files.append(file)

    return sorted(markdown_files)


def read_markdown(file_path: Path) -> str:
    """
    Read markdown file.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()