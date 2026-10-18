import re


def split_into_sections(markdown: str):
    """
    Split markdown into logical sections based on headings.
    """

    sections = []

    current_section = {
        "heading": "Introduction",
        "level": 0,
        "content": []
    }

    lines = markdown.splitlines()

    for line in lines:

        heading = re.match(r"^(#{1,6})\s+(.*)", line)

        if heading:

            if current_section["content"]:

                current_section["content"] = "\n".join(
                    current_section["content"]
                ).strip()

                sections.append(current_section)

            current_section = {
                "heading": heading.group(2),
                "level": len(heading.group(1)),
                "content": []
            }

        else:

            current_section["content"].append(line)

    if current_section["content"]:

        current_section["content"] = "\n".join(
            current_section["content"]
        ).strip()

        sections.append(current_section)

    return sections
