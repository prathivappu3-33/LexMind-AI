import yaml
import re


def extract_metadata(markdown_text: str):
    """
    Extract YAML front matter and return:
    metadata (dict)
    content (markdown without YAML)
    """

    metadata = {}

    pattern = r"^---\n(.*?)\n---\n"

    match = re.match(pattern, markdown_text, re.DOTALL)

    if match:

        yaml_text = match.group(1)

        metadata = yaml.safe_load(yaml_text)

        markdown_text = markdown_text[match.end():]

    return metadata, markdown_text