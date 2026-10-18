from gliner import GLiNER


class EntityExtractor:
    """
    Extracts entities from technical documentation using GLiNER.
    """

    def __init__(self):
        print("Loading GLiNER model...")

        self.model = GLiNER.from_pretrained(
            "urchade/gliner_medium-v2.1"
        )

        print("GLiNER model loaded successfully!")

        self.labels = [
            "Software",
            "Technical Concept",
            "CLI Command",
            "API",
            "Configuration",
            "Container",
            "Image",
            "Service",
            "Security Feature",
            "Authentication Method",
            "Environment Variable",
            "File",
            "Directory",
            "Port",
            "Protocol",
            "Programming Language",
            "Framework",
            "Tool",
            "Error",
            "Warning",
            "Best Practice",
        ]

    def extract_entities(
        self,
        text: str,
        threshold: float = 0.5,
    ):
        """
        Extract entities from text.
        """

        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=threshold,
        )

        return entities 