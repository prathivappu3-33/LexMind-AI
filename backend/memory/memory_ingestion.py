"""
Memory Ingestion Pipeline

Converts important conversations into
long-term semantic memory stored in Neo4j.
"""

from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.graph_builder import GraphBuilder
from backend.knowledge_graph.relation_extractor import RelationExtractor
from backend.memory.conversation_summarizer import ConversationSummarizer
from backend.memory.importance_scorer import ImportanceScorer


class MemoryIngestionPipeline:
    """
    Converts conversations into graph knowledge.
    """

    def __init__(self):

        self.summarizer = ConversationSummarizer()

        self.importance_scorer = ImportanceScorer()

        self.entity_extractor = EntityExtractor()

        self.relation_extractor = RelationExtractor()

        self.graph_builder = GraphBuilder()

    def ingest(
        self,
        messages,
    ):
        """
        Convert conversation into long-term memory.
        """

        #
        # Summarize conversation
        #
        summary = self.summarizer.summarize(messages)

        #
        # Score importance
        #
        score, should_store = (
            self.importance_scorer.score(summary)
        )

        #
        # Ignore trivial conversations
        #
        if not should_store:

            return {
                "stored": False,
                "importance_score": score,
                "summary": summary,
                "entities": [],
                "relations": [],
            }

        #
        # Entity extraction
        #
        entities = self.entity_extractor.extract_entities(
            summary
        )

        #
        # Relation extraction
        #
        relations = self.relation_extractor.extract(
            summary
        )

        #
        # Store in Neo4j
        #
        if relations:
            self.graph_builder.build(relations)

        return {
            "stored": True,
            "importance_score": score,
            "summary": summary,
            "entities": entities,
            "relations": relations,
        }

    def close(self):

        self.graph_builder.close()