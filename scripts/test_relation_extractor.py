from backend.knowledge_graph.relation_extractor import RelationExtractor

extractor = RelationExtractor()

texts = [
    "Docker Hub supports multi-factor authentication.",
    "Git uses SHA-1 hashing.",
    "Kubernetes uses etcd as its distributed key-value store.",
    "ROS2 uses DDS for communication.",
    "Amazon Web Services provides EC2 and S3.",
]

for text in texts:
    print("=" * 80)
    print("INPUT:")
    print(text)

    relations = extractor.extract(text)

    print("\nEXTRACTED RELATIONS:")
    if not relations:
        print("No relations found.")
    else:
        for relation in relations:
            print(relation)

    print()