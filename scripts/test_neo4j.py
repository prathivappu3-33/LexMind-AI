from backend.knowledge_graph.neo4j_client import Neo4jClient

client = Neo4jClient()

print(client.test_connection())

client.close()