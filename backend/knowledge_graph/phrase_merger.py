from typing import List, Dict


class PhraseMerger:
    """
    Merges adjacent entities into a single technical phrase.

    Example:
        docker + login  -> docker login
        Amazon + Web + Services -> Amazon Web Services
    """

    def merge(self, text: str, entities: List[Dict]) -> List[Dict]:

        if not entities:
            return []

        # Sort entities by their starting position
        entities = sorted(entities, key=lambda x: x["start"])

        merged = []

        current = entities[0].copy()

        for next_entity in entities[1:]:

            # Characters between current entity and next entity
            gap = text[current["end"]:next_entity["start"]]

            # Merge if only whitespace exists between them
            if gap.strip() == "":

                current["text"] += gap + next_entity["text"]
                current["end"] = next_entity["end"]

                # Average confidence
                current["score"] = (
                    current["score"] + next_entity["score"]
                ) / 2

                current["label"] = "Technical Phrase"

            else:

                merged.append(current)
                current = next_entity.copy()

        merged.append(current)

        return merged