import uuid
from pathlib import Path

from backend.preprocessing.reader import read_markdown
from backend.preprocessing.metadata import extract_metadata
from backend.preprocessing.cleaner import clean_markdown
from backend.preprocessing.section_splitter import split_into_sections
from backend.preprocessing.writer import save_document


def process_document(file_path: Path):
    """
    Complete preprocessing pipeline for one markdown document.
    """

    # Read markdown file
    markdown = read_markdown(file_path)

    # Extract YAML metadata
    metadata, content = extract_metadata(markdown)

    # Clean markdown
    content = clean_markdown(content)

    # Split into logical sections
    sections = split_into_sections(content)

    # Create structured document
    document = {
        "document_id": str(uuid.uuid4()),

        "source": str(file_path),

        "title": metadata.get("title", file_path.stem),

        "metadata": metadata,

        "sections": sections,

        "statistics": {
            "num_sections": len(sections),
            "num_words": len(content.split()),
            "num_characters": len(content)
        }
    }

    # Save processed document
    save_document(document)

    return document