from backend.knowledge_graph.entity_extractor import EntityExtractor


text = """
Run docker login to authenticate with Docker Hub.

Recovery codes can be generated from the Account Settings page.

Docker supports Multi-Factor Authentication.
"""

extractor = EntityExtractor()

entities = extractor.extract_entities(text)

print("\nDetected Entities\n")

for entity in entities:
    print(entity)