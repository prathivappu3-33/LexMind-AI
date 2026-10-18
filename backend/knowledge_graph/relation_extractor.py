"""
Relation Extraction Module

Uses the REBEL model to extract knowledge graph triples from text.

Output Format:
[
    {
        "head": "...",
        "relation": "...",
        "tail": "..."
    }
]
"""

from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RelationExtractor:
    """
    Extracts knowledge graph relations using the REBEL model.
    """

    MODEL_NAME = "Babelscape/rebel-large"

    def __init__(self):

        print("Loading REBEL model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME
        )

        print("REBEL model loaded successfully!\n")

    def extract(self, text: str) -> List[Dict]:
        """
        Extract relations from input text.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        generated_tokens = self.model.generate(
            **inputs,
            max_length=256,
        )

        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=False,
        )[0]

        return self._parse_output(generated_text)

    def _parse_output(
        self,
        generated_text: str
    ) -> List[Dict]:
        """
        Parse REBEL output into clean triples.

        A valid REBEL triplet must have:

            <triplet> HEAD <subj> TAIL <obj> RELATION

        Malformed triplets containing repeated <subj>
        or <obj> markers are discarded.
        """

        generated_text = (
            generated_text
            .replace("<s>", "")
            .replace("</s>", "")
            .strip()
        )

        if not generated_text:
            return []

        # --------------------------------------------------
        # Split output into individual triplet sections
        # --------------------------------------------------

        raw_triplets = generated_text.split("<triplet>")

        relations = []

        for raw_triplet in raw_triplets:

            raw_triplet = raw_triplet.strip()

            if not raw_triplet:
                continue

            # --------------------------------------------------
            # A valid triplet must contain exactly one
            # <subj> and exactly one <obj>
            # --------------------------------------------------

            if raw_triplet.count("<subj>") != 1:
                continue

            if raw_triplet.count("<obj>") != 1:
                continue

            try:

                head_part, remaining = raw_triplet.split(
                    "<subj>",
                    1
                )

                tail_part, relation_part = remaining.split(
                    "<obj>",
                    1
                )

            except ValueError:
                # Malformed REBEL output
                continue

            head = head_part.strip()
            tail = tail_part.strip()
            relation = relation_part.strip()

            # --------------------------------------------------
            # Validate fields
            # --------------------------------------------------

            if not head:
                continue

            if not tail:
                continue

            if not relation:
                continue

            # --------------------------------------------------
            # Remove obvious duplicate token sequences
            # --------------------------------------------------

            relation_words = relation.split()

            if len(relation_words) > 1:
                half = len(relation_words) // 2

                if (
                    len(relation_words) % 2 == 0
                    and relation_words[:half]
                    == relation_words[half:]
                ):
                    relation = " ".join(
                        relation_words[:half]
                    )

            tail_words = tail.split()

            if len(tail_words) > 1:
                half = len(tail_words) // 2

                if (
                    len(tail_words) % 2 == 0
                    and tail_words[:half]
                    == tail_words[half:]
                ):
                    tail = " ".join(
                        tail_words[:half]
                    )

            # --------------------------------------------------
            # Remove self-relations
            # --------------------------------------------------

            if head.lower() == tail.lower():
                continue

            relations.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                }
            )

        return relations