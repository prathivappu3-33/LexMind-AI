from typing import List, Dict


class EntityPostProcessor:
    """
    Cleans and normalizes entities extracted by GLiNER.
    """

    def __init__(self):
        # Words we don't want as standalone entities
        self.stop_entities = {
            "run",
            "click",
            "open",
            "use",
            "create",
            "delete",
            "install",
            "start",
            "stop",
            "page",
        }

    def process(self, entities: List[Dict]) -> List[Dict]:

        cleaned = []

        seen = set()

        for entity in entities:

            text = entity["text"].strip()

            label = entity["label"]

            # Remove unwanted words
            if text.lower() in self.stop_entities:
                continue

            # Normalize "Docker"
            if text.lower() == "docker":
                text = "Docker"

            # Normalize Account Settings page
            if text.lower() == "account settings page":
                text = "Account Settings"

            key = (text, label)

            if key in seen:
                continue

            seen.add(key)

            cleaned.append(
                {
                    "text": text,
                    "label": label,
                    "score": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"],
                }
            )

        return cleaned