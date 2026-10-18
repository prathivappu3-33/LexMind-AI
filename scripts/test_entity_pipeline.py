from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.entity_postprocessor import EntityPostProcessor
from backend.knowledge_graph.phrase_merger import PhraseMerger


TEST_CASES = [
    (
        "Docker",
        "Run docker login to authenticate with Docker Hub."
    ),
    (
        "Git",
        "Clone the repository using git clone and push changes using git push."
    ),
    (
        "Kubernetes",
        "Deploy the application using kubectl apply."
    ),
    (
        "Python",
        "Install packages using pip install transformers."
    ),
    (
        "ROS2",
        "Launch the robot using ros2 launch turtlebot4_bringup."
    ),
    (
        "AWS",
        "Amazon Web Services provides EC2 and S3."
    ),
    (
        "Linux",
        "Create a directory using mkdir and navigate using cd."
    ),
    (
        "Terraform",
        "Initialize the infrastructure using terraform init."
    ),
]


extractor = EntityExtractor()
postprocessor = EntityPostProcessor()
merger = PhraseMerger()

for domain, text in TEST_CASES:

    print("\n" + "=" * 80)
    print(f"DOMAIN : {domain}")
    print("=" * 80)

    print("\nINPUT TEXT\n")
    print(text)

    # -------------------------------------------------------
    # Step 1 : Entity Extraction
    # -------------------------------------------------------

    raw_entities = extractor.extract_entities(text)

    print("\nRAW ENTITIES\n")

    for entity in raw_entities:
        print(entity)

    # -------------------------------------------------------
    # Step 2 : Post Processing
    # -------------------------------------------------------

    clean_entities = postprocessor.process(raw_entities)

    print("\nPOST PROCESSED ENTITIES\n")

    for entity in clean_entities:
        print(entity)

    # -------------------------------------------------------
    # Step 3 : Phrase Merging
    # -------------------------------------------------------

    merged_entities = merger.merge(text, clean_entities)

    print("\nMERGED ENTITIES\n")

    for entity in merged_entities:
        print(entity)

print("\nPipeline Test Completed Successfully!")