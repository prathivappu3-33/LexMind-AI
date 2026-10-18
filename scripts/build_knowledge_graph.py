"""
Build Knowledge Graph

Reads every chunk from datasets/chunks and builds
the complete Neo4j Knowledge Graph.
"""

import json
from pathlib import Path

from backend.knowledge_graph.processor import KnowledgeGraphProcessor


def main():

    # --------------------------------------------------
    # Locate all chunk files
    # --------------------------------------------------

    chunks_dir = Path("datasets/chunks")

    chunk_files = sorted(chunks_dir.rglob("*.json"))

    print(f"\nFound {len(chunk_files)} chunk files.\n")

    if len(chunk_files) == 0:
        print("No chunk files found.")
        return

    # --------------------------------------------------
    # Initialize KG Pipeline
    # --------------------------------------------------

    kg = KnowledgeGraphProcessor()

    print("\nClearing existing Neo4j graph...\n")

    kg.clear_graph()

    # --------------------------------------------------
    # Process every chunk
    # --------------------------------------------------

    processed = 0
    failed = 0

    for i, chunk_file in enumerate(chunk_files, start=1):

        try:

            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk = json.load(f)

            text = chunk.get("text", "").strip()

            if not text:
                print(f"[{i}/{len(chunk_files)}] Empty chunk skipped.")
                continue

            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source": chunk.get("source"),
                "title": chunk.get("title"),
                "section_heading": chunk.get("section_heading"),
            }

            kg.process(
                text,
                metadata=metadata,
            )

            processed += 1

            print(
                f"[{i}/{len(chunk_files)}] "
                f"Processed: {chunk_file.name}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[{i}/{len(chunk_files)}] "
                f"FAILED: {chunk_file.name}"
            )

            print(e)

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    kg.close()

    print("\n========================================")
    print("Knowledge Graph Build Complete")
    print("========================================")
    print(f"Processed : {processed}")
    print(f"Failed    : {failed}")
    print("========================================")


if __name__ == "__main__":
    main()