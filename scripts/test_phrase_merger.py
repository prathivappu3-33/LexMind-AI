from backend.knowledge_graph.phrase_merger import PhraseMerger


merger = PhraseMerger()


text = (
    "Amazon Web Services provides EC2. "
    "Run docker login to authenticate with Docker Hub. "
    "Deploy using kubectl apply."
)

entities = [
    {
        "text": "Amazon",
        "start": 0,
        "end": 6,
        "label": "Organization",
        "score": 0.95,
    },
    {
        "text": "Web",
        "start": 7,
        "end": 10,
        "label": "Organization",
        "score": 0.94,
    },
    {
        "text": "Services",
        "start": 11,
        "end": 19,
        "label": "Organization",
        "score": 0.96,
    },
    {
        "text": "docker",
        "start": 37,
        "end": 43,
        "label": "Container",
        "score": 0.91,
    },
    {
        "text": "login",
        "start": 44,
        "end": 49,
        "label": "CLI Command",
        "score": 0.90,
    },
    {
        "text": "Docker Hub",
        "start": 66,
        "end": 76,
        "label": "Service",
        "score": 0.92,
    },
    {
        "text": "kubectl",
        "start": 92,
        "end": 100,
        "label": "CLI Command",
        "score": 0.94,
    },
    {
        "text": "apply",
        "start": 101,
        "end": 106,
        "label": "CLI Command",
        "score": 0.93,
    },
]

print("\nOriginal Entities\n")

for entity in entities:
    print(entity)

merged = merger.merge(text, entities)

print("\nMerged Entities\n")

for entity in merged:
    print(entity)