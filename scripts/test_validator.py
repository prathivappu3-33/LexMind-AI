from backend.evaluation.validator import BenchmarkValidator
from backend.llm.gemma_client import GemmaClient


def main():
    llm = GemmaClient()

    validator = BenchmarkValidator(
        llm=llm,
        candidate_path="datasets/ground_truth/candidate_ground_truth.json",
        output_path="datasets/ground_truth",
        confidence_threshold=0.8,
    )

    validator.run()


if __name__ == "__main__":
    main()