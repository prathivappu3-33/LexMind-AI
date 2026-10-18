"""
Ground Truth Generator

Generates candidate ground-truth chunk IDs for retrieval evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from backend.retrieval.vector_retriever import VectorRetriever


class GroundTruthGenerator:
    """
    Generates candidate ground-truth chunks from benchmark questions.
    """

    def __init__(
        self,
        questions_path: str = "datasets/questions/docker_questions.json",
        output_path: str = "datasets/ground_truth/docker_ground_truth.json",
        top_k: int = 10,
        min_score: float = 0.70,
    ):

        self.questions_path = Path(questions_path)
        self.output_path = Path(output_path)

        self.top_k = top_k
        self.min_score = min_score

        print("\nInitializing Vector Retriever...\n")

        self.retriever = VectorRetriever()

        print("\nGround Truth Generator Ready!\n")

    # -----------------------------------------------------
    # Question Loader
    # -----------------------------------------------------

    def load_questions(self) -> List[Dict]:

        if not self.questions_path.exists():
            raise FileNotFoundError(
                f"Questions file not found: {self.questions_path}"
            )

        with open(
            self.questions_path,
            "r",
            encoding="utf-8",
        ) as f:
            questions = json.load(f)

        print(f"Loaded {len(questions)} questions.\n")

        return questions

    # -----------------------------------------------------
    # Candidate Generation
    # -----------------------------------------------------

    def generate_candidates(self) -> List[Dict]:

        questions = self.load_questions()

        all_candidates = []

        total = len(questions)

        for index, sample in enumerate(questions, start=1):

            question = sample["question"]
            question_id = sample["id"]

            print("=" * 80)
            print(f"[{index}/{total}] {question}")
            print("=" * 80)

            try:

                results = self.retriever.retrieve(
                    query=question,
                    top_k=self.top_k,
                    min_score=self.min_score,
                )


            except Exception as e:

                print(f"Retrieval failed: {e}")

                all_candidates.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "candidate_chunks": [],
                        "status": "FAILED",
                    }
                )

                continue

            candidates = []

            for result in results:

                candidate = {

                    "chunk_id": result.get("chunk_id"),

                    "document_id": result.get("document_id"),

                    "source": result.get("source"),

                    "title": result.get("title"),

                    "section_heading": result.get("section_heading"),

                    "section_level": result.get("section_level"),

                    "chunk_index": result.get("chunk_index"),

                    "score": round(
                        float(result.get("score", 0.0)),
                        4
                    ),

                    "preview": result.get(
                        "text",
                        ""
                    )[:300]

                }

                candidates.append(candidate)

            print(f"Retrieved {len(candidates)} candidate chunks.\n")

            all_candidates.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "candidate_chunks": candidates,
                    "status": "SUCCESS",
                }
            )

        return all_candidates

    # -----------------------------------------------------
    # Save Results
    # -----------------------------------------------------

    def save(self, data: List[Dict]) -> None:

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print("\n" + "=" * 80)
        print("Ground Truth Generation Complete")
        print("=" * 80)
        print(f"Questions Processed : {len(data)}")
        print(f"Output File         : {self.output_path}")
        print("=" * 80)

    # -----------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------

    def run(self) -> None:

        print("\nStarting Ground Truth Generation...\n")

        candidates = self.generate_candidates()

        self.save(candidates)

        print("\nGround truth generation finished successfully.\n")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

def main():

    generator = GroundTruthGenerator(
        questions_path="datasets/questions/docker_questions.json",
        output_path="datasets/ground_truth/docker_ground_truth.json",
        top_k=10,
        min_score=0.70,
    )

    generator.run()


if __name__ == "__main__":
    main()