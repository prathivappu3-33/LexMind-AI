import re


def clean_markdown(text: str):
    """
    Clean markdown while preserving useful information.
    """

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove Hugo shortcodes
    text = re.sub(r"\{\{<.*?>\}\}", "", text)
    text = re.sub(r"\{\{%.*?%\}\}", "", text)

    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Convert markdown links
    # [Docker](https://docs.docker.com)
    # becomes
    # Docker
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # Normalize blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()