"""
Relation Validator

Cleans and normalizes relations produced by REBEL before
they are inserted into the Knowledge Graph.
"""

from typing import Dict, List


class RelationValidator:
    """
    Cleans, normalizes, and deduplicates REBEL relations.
    """

    # --------------------------------------------------
    # Canonical relation mapping
    # --------------------------------------------------

    RELATION_MAP = {

        # Usage
        "use": "USES",
        "uses": "USES",
        "used by": "USES",

        # Development
        "developer": "DEVELOPS",
        "develops": "DEVELOPS",

        # Production
        "product or material produced": "PRODUCES",
        "produces": "PRODUCES",

        # Ontology
        "instance of": "INSTANCE_OF",
        "subclass of": "SUBCLASS_OF",

        # Composition
        "part of": "PART_OF",
        "has part": "HAS_PART",

        # Dependency / architecture
        "based on": "BASED_ON",

        # Other common relations
        "depends on": "DEPENDS_ON",
        "requires": "REQUIRES",
        "contains": "CONTAINS",
        "created by": "CREATED_BY",
        "provided by": "PROVIDED_BY",
        "supports": "SUPPORTS",
        "runs on": "RUNS_ON",
        "different from": "DIFFERENT_FROM",
    }

    def validate(self, relations: List[Dict]) -> List[Dict]:

        validated = []

        seen = set()

        for relation in relations:

            # --------------------------------------------------
            # Safely extract fields
            # --------------------------------------------------

            head = str(relation.get("head", "")).strip()
            rel = str(relation.get("relation", "")).strip().lower()
            tail = str(relation.get("tail", "")).strip()

            # --------------------------------------------------
            # Remove empty triples
            # --------------------------------------------------

            if not head or not rel or not tail:
                continue

            # --------------------------------------------------
            # Normalize whitespace
            # --------------------------------------------------

            rel = " ".join(rel.split())

            # --------------------------------------------------
            # Remove obvious duplicated relation phrases
            # --------------------------------------------------

            duplicate_patterns = [
                "product or material produced product or material produced",
                "part of part of",
                "has part has part",
                "based on based on",
                "uses uses",
                "use use",
                "used by used by",
            ]

            for pattern in duplicate_patterns:

                if rel == pattern:

                    rel = pattern.split(
                        " " + pattern.split(" ")[0]
                    )[0]

                    break

            # --------------------------------------------------
            # Explicit cleanup for duplicated phrases
            # --------------------------------------------------

            if rel == "product or material produced product or material produced":
                rel = "product or material produced"

            elif rel == "part of part of":
                rel = "part of"

            elif rel == "has part has part":
                rel = "has part"

            elif rel == "based on based on":
                rel = "based on"

            elif rel == "uses uses":
                rel = "uses"

            elif rel == "used by used by":
                rel = "used by"

            # --------------------------------------------------
            # Handle inverse relation: "used by"
            #
            # A uses B
            # B is used by A
            #
            # Normalize both to:
            #
            # A --USES--> B
            # --------------------------------------------------

            if rel == "used by":

                head, tail = tail, head
                rel = "uses"

            # --------------------------------------------------
            # Map to canonical relation
            # --------------------------------------------------

            rel = self.RELATION_MAP.get(
                rel,
                rel.upper()
            )

            # --------------------------------------------------
            # Ensure Neo4j-safe relationship name
            # --------------------------------------------------

            rel = rel.replace(" ", "_")

            # --------------------------------------------------
            # Remove self-loops
            # --------------------------------------------------

            if head.lower() == tail.lower():
                continue

            # --------------------------------------------------
            # Duplicate detection
            # --------------------------------------------------

            key = (
                head.lower(),
                rel,
                tail.lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            # --------------------------------------------------
            # Store validated triple
            # --------------------------------------------------

            validated.append(
                {
                    "head": head,
                    "relation": rel,
                    "tail": tail,
                }
            )

        return validated