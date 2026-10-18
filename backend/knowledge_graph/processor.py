"""
Knowledge Graph Processor

Pipeline

Text
    │
    ▼
Entity Extraction
    ▼
Entity Postprocessing
    ▼
Phrase Merging
    ▼
Relation Extraction
    ▼
Relation Validation
    ▼
Graph Builder
"""

from typing import Dict

from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.entity_postprocessor import EntityPostProcessor
from backend.knowledge_graph.phrase_merger import PhraseMerger
from backend.knowledge_graph.relation_extractor import RelationExtractor
from backend.knowledge_graph.relation_validator import RelationValidator
from backend.knowledge_graph.graph_builder import GraphBuilder


class KnowledgeGraphProcessor:

    def __init__(self):

        print("\nInitializing Knowledge Graph Pipeline...\n")

        self.entity_extractor = EntityExtractor()

        self.entity_postprocessor = EntityPostProcessor()

        self.phrase_merger = PhraseMerger()

        self.relation_extractor = RelationExtractor()

        self.relation_validator = RelationValidator()

        self.graph_builder = GraphBuilder()

        print("\nKnowledge Graph Pipeline Ready!\n")

    def process(
    self,
    text: str,
    metadata: Dict | None = None,
) -> Dict:

        # -----------------------------
        # Entity Pipeline
        # -----------------------------

        raw_entities = self.entity_extractor.extract_entities(text)

        clean_entities = self.entity_postprocessor.process(
            raw_entities
        )

        merged_entities = self.phrase_merger.merge(
            text,
            clean_entities
        )

        # -----------------------------
        # Relation Pipeline
        # -----------------------------

        raw_relations = self.relation_extractor.extract(
            text
        )

        validated_relations = self.relation_validator.validate(
            raw_relations
        )

        # -----------------------------
        # Build Graph
        # -----------------------------

        provenance = metadata or {}

        provenance["evidence"] = text

        self.graph_builder.build(
            validated_relations,
            provenance=provenance,
        )

        return {
            "text": text,
            "entities": merged_entities,
            "relations": validated_relations,
        }

    def clear_graph(self):

        self.graph_builder.clear()

    def close(self):

        self.graph_builder.close()