from backend.retrieval.query_processor import QueryProcessor

processor = QueryProcessor()

queries = [

    "How does Docker Hub implement multi-factor authentication?",

    "How do I create a Docker container?",

    "How does ROS2 communicate using DDS?",

    "How can I configure AWS EC2 networking?"

]

for query in queries:

    print("=" * 80)

    result = processor.process(query)

    print("QUERY")
    print(result["original_query"])

    print("\nENTITIES")

    for entity in result["entities"]:
        print(entity)

    print("\nEMBEDDING SHAPE")
    print(result["embedding"].shape)

    print()