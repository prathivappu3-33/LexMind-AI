from backend.knowledge_graph.graph_builder import GraphBuilder

builder = GraphBuilder()

# Uncomment if you want a fresh graph each time
builder.clear()

relations = [

    {
        "head": "Git",
        "relation": "USES",
        "tail": "SHA-1",
    },

    {
        "head": "Docker Hub",
        "relation": "USES",
        "tail": "Authentication",
    },

    {
        "head": "Amazon Web Services",
        "relation": "PRODUCES",
        "tail": "EC2",
    },

    {
        "head": "Amazon Web Services",
        "relation": "PRODUCES",
        "tail": "S3",
    },

]

builder.build(relations)

print("\nKnowledge Graph created successfully!\n")

builder.close()