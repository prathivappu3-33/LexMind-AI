from neo4j import GraphDatabase

from backend.config import settings, get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Handles all communication with Neo4j.
    """

    def __init__(self):
        logger.info("Connecting to Neo4j...")

        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USER,
                settings.NEO4J_PASSWORD,
            ),
        )

        logger.info("Successfully connected to Neo4j.")

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            logger.info("Closing Neo4j connection.")
            self.driver.close()

    def execute_query(self, query, parameters=None):
        parameters = parameters or {}

        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def test_connection(self):
        query = """
        RETURN "LexMindAI Connected!" AS message
        """

        result = self.execute_query(query)

        return result[0]["message"]

    # ------------------------------------------------------------------
    # Node Operations
    # ------------------------------------------------------------------

    def create_entity(self, name: str):

        query = """
        MERGE (e:Entity {name:$name})
        """

        self.execute_query(
            query,
            {
                "name": name
            }
        )

    # ------------------------------------------------------------------
    # Relationship Operations
    # ------------------------------------------------------------------

    def create_relationship(
        self,
        head: str,
        relation: str,
        tail: str,
        provenance=None,
    ):
        """
        Create or update a relationship with provenance.

        Multiple chunks may support the same relationship, so
        provenance is stored as lists on the relationship.
        """

        provenance = provenance or {}

        chunk_id = provenance.get("chunk_id")
        document_id = provenance.get("document_id")
        source = provenance.get("source")
        title = provenance.get("title")
        section_heading = provenance.get("section_heading")
        evidence = provenance.get("evidence")

        query = f"""
        MERGE (h:Entity {{name: $head}})
        MERGE (t:Entity {{name: $tail}})
        MERGE (h)-[r:{relation}]->(t)

        ON CREATE SET
            r.chunk_ids = CASE
                WHEN $chunk_id IS NULL THEN []
                ELSE [$chunk_id]
            END,
            r.document_ids = CASE
                WHEN $document_id IS NULL THEN []
                ELSE [$document_id]
            END,
            r.sources = CASE
                WHEN $source IS NULL THEN []
                ELSE [$source]
            END,
            r.titles = CASE
                WHEN $title IS NULL THEN []
                ELSE [$title]
            END,
            r.sections = CASE
                WHEN $section_heading IS NULL THEN []
                ELSE [$section_heading]
            END,
            r.evidence = CASE
                WHEN $evidence IS NULL THEN []
                ELSE [$evidence]
            END

        ON MATCH SET
            r.chunk_ids =
                CASE
                    WHEN $chunk_id IS NULL
                        OR $chunk_id IN coalesce(r.chunk_ids, [])
                    THEN coalesce(r.chunk_ids, [])
                    ELSE coalesce(r.chunk_ids, []) + $chunk_id
                END,

            r.document_ids =
                CASE
                    WHEN $document_id IS NULL
                        OR $document_id IN coalesce(r.document_ids, [])
                    THEN coalesce(r.document_ids, [])
                    ELSE coalesce(r.document_ids, []) + $document_id
                END,

            r.sources =
                CASE
                    WHEN $source IS NULL
                        OR $source IN coalesce(r.sources, [])
                    THEN coalesce(r.sources, [])
                    ELSE coalesce(r.sources, []) + $source
                END,

            r.titles =
                CASE
                    WHEN $title IS NULL
                        OR $title IN coalesce(r.titles, [])
                    THEN coalesce(r.titles, [])
                    ELSE coalesce(r.titles, []) + $title
                END,

            r.sections =
                CASE
                    WHEN $section_heading IS NULL
                        OR $section_heading IN coalesce(r.sections, [])
                    THEN coalesce(r.sections, [])
                    ELSE coalesce(r.sections, []) + $section_heading
                END,

            r.evidence =
                CASE
                    WHEN $evidence IS NULL
                        OR $evidence IN coalesce(r.evidence, [])
                    THEN coalesce(r.evidence, [])
                    ELSE coalesce(r.evidence, []) + $evidence
                END
        """

        self.execute_query(
            query,
            {
                "head": head,
                "tail": tail,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "title": title,
                "section_heading": section_heading,
                "evidence": evidence,
            },
        )

    # ------------------------------------------------------------------
    # Graph Operations
    # ------------------------------------------------------------------

    def clear_graph(self):

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        self.execute_query(query)

    # ------------------------------------------------------------------
    # Graph Retrieval
    # ------------------------------------------------------------------

    def find_entity(self, name: str):

        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name)=toLower($name)
        RETURN e.name AS entity
        """

        return self.execute_query(
            query,
            {
                "name": name
            }
        )

    def search_entities(self, text: str, limit: int = 10):
        """
        Find graph entities whose names appear in the input text.

        Uses case-insensitive substring matching.
        Longer entity names are preferred by Neo4j ordering.
        """

        query = """
        MATCH (e:Entity)
        WHERE toLower($text) CONTAINS toLower(e.name)

        RETURN
            e.name AS entity

        ORDER BY size(e.name) DESC

        LIMIT $limit
        """

        return self.execute_query(
            query,
            {
                "text": text,
                "limit": limit,
            }
        )

    def get_neighbors(self, entity_name: str):

        query = """
        MATCH (e:Entity)-[r]-(n:Entity)

        WHERE toLower(e.name)=toLower($entity)

        RETURN
            e.name AS source,
            type(r) AS relation,
            n.name AS target
        """

        return self.execute_query(
            query,
            {
                "entity": entity_name
            }
        )

    def get_subgraph(
        self,
        entity_name: str,
        max_hops: int = 2,
        limit: int = 20
    ):
        """
        Retrieve a subgraph together with relationship provenance.
        """

        query = f"""
        MATCH path=(e:Entity)-[*1..{max_hops}]-(n:Entity)

        WHERE toLower(e.name)=toLower($entity)

        UNWIND relationships(path) AS rel

        WITH DISTINCT
            startNode(rel).name AS source,
            type(rel) AS relation,
            endNode(rel).name AS target,
            rel.chunk_ids AS chunk_ids,
            rel.document_ids AS document_ids,
            rel.sources AS sources,
            rel.titles AS titles,
            rel.sections AS sections,
            rel.evidence AS evidence

        RETURN
            source,
            relation,
            target,
            chunk_ids,
            document_ids,
            sources,
            titles,
            sections,
            evidence

        LIMIT $limit
        """

        return self.execute_query(
            query,
            {
                "entity": entity_name,
                "limit": limit
            }
        )

