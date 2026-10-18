from backend.knowledge_graph.processor import KnowledgeGraphProcessor

processor = KnowledgeGraphProcessor()

processor.clear_graph()

texts = [

    "Git uses SHA-1 hashing.",

    "Docker Hub supports multi-factor authentication.",

    "Amazon Web Services provides EC2 and S3.",

    "ROS2 uses DDS for communication.",

]

for text in texts:

    print("=" * 80)

    result = processor.process(text)

    print("TEXT")
    print(result["text"])

    print("\nENTITIES")

    for entity in result["entities"]:
        print(entity)

    print("\nRELATIONS")

    for relation in result["relations"]:
        print(relation)

processor.close()

print("\nKnowledge Graph Pipeline Completed Successfully!\n")