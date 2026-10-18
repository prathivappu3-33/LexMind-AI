from backend.preprocessing.reader import find_markdown_files
from backend.preprocessing.processor import process_document


def main():

    markdown_files = find_markdown_files()

    print(f"\nFound {len(markdown_files)} markdown files.\n")

    for index, file_path in enumerate(markdown_files, start=1):

        process_document(file_path)

        print(
            f"[{index}/{len(markdown_files)}] Processed: {file_path.name}"
        )

    print("\n")
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(
        f"Successfully processed {len(markdown_files)} documents."
    )


if __name__ == "__main__":
    main()