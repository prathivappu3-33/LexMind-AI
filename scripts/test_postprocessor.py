from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.entity_postprocessor import EntityPostProcessor


text = """
Run docker login to authenticate with Docker Hub.

Recovery codes can be generated from the Account Settings page.

Docker supports Multi-Factor Authentication.
"""

extractor = EntityExtractor()

processor = EntityPostProcessor()

entities = extractor.extract_entities(text)

clean_entities = processor.process(entities)

print("\nCleaned Entities\n")

for entity in clean_entities:
    print(entity)