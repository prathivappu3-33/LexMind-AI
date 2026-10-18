from pathlib import Path
import json

from backend.chunking.processor import process_chunks


PROCESSED_DIR = Path("datasets/processed_documents")


def main():

    json_files = sorted(PROCESSED_DIR.rglob("*.json"))

    print(f"\nFound {len(json_files)} processed documents.\n")

    total_chunks = 0

    for index, file in enumerate(json_files, start=1):

        with open(file, "r", encoding="utf-8") as f:
            document = json.load(f)

        chunks = process_chunks(document)

        total_chunks += len(chunks)

        print(
            f"[{index}/{len(json_files)}] "
            f"{file.name} -> {len(chunks)} chunks"
        )

    print("\n" + "=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)

    print(f"Processed Documents : {len(json_files)}")
    print(f"Total Chunks        : {total_chunks}")


if __name__ == "__main__":
    main()