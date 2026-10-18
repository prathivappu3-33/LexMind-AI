"""
Graph Builder

Converts validated relations into a Neo4j Knowledge Graph
with source provenance.
"""

from typing import Dict, List

from backend.knowledge_graph.neo4j_client import Neo4jClient


class GraphBuilder:
    """
    Builds the Knowledge Graph inside Neo4j.
    """

    def __init__(self):

        self.neo4j = Neo4jClient()

    def build(
        self,
        relations: List[Dict],
        provenance: Dict | None = None,
    ) -> None:
        """
        Insert validated relations into Neo4j.

        Provenance records where each relation came from.
        """

        provenance = provenance or {}

        for relation in relations:

            head = relation["head"]
            rel = relation["relation"]
            tail = relation["tail"]

            self.neo4j.create_entity(head)
            self.neo4j.create_entity(tail)

            self.neo4j.create_relationship(
                head=head,
                relation=rel,
                tail=tail,
                provenance=provenance,
            )

    def clear(self):
        """
        Delete the complete graph.
        """

        self.neo4j.clear_graph()

    def close(self):
        """
        Close Neo4j connection.
        """

        self.neo4j.close()
