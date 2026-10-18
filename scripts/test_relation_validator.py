from backend.knowledge_graph.relation_validator import RelationValidator

validator = RelationValidator()

relations = [

    {
        "head": "Git",
        "relation": "uses",
        "tail": "SHA-1"
    },

    {
        "head": "SHA-1",
        "relation": "used by",
        "tail": "Git"
    },

    {
        "head": "Docker",
        "relation": "use",
        "tail": "Docker"
    },

    {
        "head": "Amazon Web Services",
        "relation": "product or material produced",
        "tail": "EC2"
    },

    {
        "head": "Amazon Web Services",
        "relation": "product or material produced",
        "tail": "EC2"
    }

]

validated = validator.validate(relations)

print("\nVALIDATED RELATIONS\n")

for relation in validated:
    print(relation)