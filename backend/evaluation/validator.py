"""
Benchmark Validator

Uses an LLM to judge whether retrieved candidate chunks are
directly relevant to benchmark questions.

Outputs:
1. validation_results.json
2. ground_truth.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class BenchmarkValidator:
    """
    Validate candidate retrieval chunks and create final ground truth.
    """

    def __init__(
        self,
        llm,
        candidate_path: str,
        output_path: str,
        confidence_threshold: float = 0.80,
    ):
        self.llm = llm
        self.candidate_path = Path(candidate_path)
        self.output_path = Path(output_path)
        self.confidence_threshold = confidence_threshold

    # ---------------------------------------------------------
    # Load candidates
    # ---------------------------------------------------------

    def load_candidates(self) -> List[Dict]:

        if not self.candidate_path.exists():
            raise FileNotFoundError(
                f"Candidate file not found: {self.candidate_path}"
            )

        with open(
            self.candidate_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "Candidate ground-truth file must contain a JSON list."
            )

        return data

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        chunk: Dict,
    ) -> str:

        text = (
            chunk.get("text")
            or chunk.get("preview")
            or ""
        )

        title = chunk.get("title") or ""
        section = chunk.get("section_heading") or ""

        return f"""
You are an expert relevance judge creating ground-truth labels
for an information-retrieval benchmark.

Your job is NOT to decide whether the passage is generally related
to the topic.

Your job is to determine whether the passage contains information
that DIRECTLY helps answer the specific question.

QUESTION:
{question}

DOCUMENT TITLE:
{title}

SECTION:
{section}

PASSAGE:
{text}

STRICT RELEVANCE RULES:

Mark "relevant": true ONLY when the passage contains factual
information that directly contributes to answering the question.

Mark "relevant": false when:

1. The passage only mentions the same topic or keywords.
2. The passage is merely navigation, a heading, next steps,
   installation instructions, links, or references.
3. The passage discusses a related concept but does not answer
   the actual question.
4. The reader would still need another passage to obtain any
   meaningful part of the answer.
5. The relationship to the question is weak, indirect, or inferred
   only because both concern Docker.
6. The passage contains no substantive answer-bearing information.

IMPORTANT:

Do NOT mark a passage relevant merely because it contains words
from the question.

Prefer FALSE when uncertain.

Examples:

Question:
"What is Docker?"

Passage:
"Install Docker. Get started with Docker."

Judgment:
false

Reason:
The passage contains Docker-related navigation but does not explain
what Docker is.

Question:
"What is Docker?"

Passage:
"Docker is an open platform for developing, shipping, and running
applications."

Judgment:
true

Reason:
The passage directly defines Docker.

Question:
"What is a Docker container?"

Passage:
"A container is a runnable instance of an image."

Judgment:
true

Reason:
The passage directly defines a container.

Question:
"What is a Docker container?"

Passage:
"Docker Desktop is available for Windows, Linux, and macOS."

Judgment:
false

Reason:
The passage is about Docker Desktop platform support and does not
explain containers.

CONFIDENCE:

Use a number between 0.0 and 1.0.

0.90 - 1.00:
Very clear judgment.

0.80 - 0.89:
Strong judgment.

0.60 - 0.79:
Some uncertainty.

Below 0.60:
Highly uncertain.

Return ONLY valid JSON.

Do not include markdown fences.
Do not include text before or after the JSON.

Required format:

{{
    "relevant": true,
    "confidence": 0.95,
    "reason": "The passage directly provides information needed to answer the question."
}}
""".strip()

    # ---------------------------------------------------------
    # JSON extraction
    # ---------------------------------------------------------

    @staticmethod
    def extract_json(raw_response: str) -> Dict:

        if not raw_response:
            raise ValueError("LLM returned an empty response.")

        response = raw_response.strip()

        # Gemma may still return ```json ... ```
        if response.startswith("```"):

            lines = response.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response = "\n".join(lines).strip()

        # Extract first JSON object if extra text exists
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No JSON object found in response: {raw_response}"
            )

        json_text = response[start:end + 1]

        parsed = json.loads(json_text)

        if not isinstance(parsed, dict):
            raise ValueError(
                "LLM response JSON must be an object."
            )

        return parsed

    # ---------------------------------------------------------
    # Normalize verdict
    # ---------------------------------------------------------

    @staticmethod
    def normalize_relevant(value) -> bool:

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {
                "true",
                "yes",
                "relevant",
                "1",
            }

        return bool(value)

    @staticmethod
    def normalize_confidence(value) -> float:

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0

        return max(
            0.0,
            min(1.0, confidence),
        )

    # ---------------------------------------------------------
    # Validate one chunk
    # ---------------------------------------------------------

    def validate_chunk(
        self,
        question: str,
        chunk: Dict,
    ) -> Dict:

        chunk_id = chunk.get("chunk_id")

        try:

            prompt = self.build_prompt(
                question,
                chunk,
            )

            raw_response = self.llm.generate(prompt)

            parsed = self.extract_json(raw_response)

            relevant = self.normalize_relevant(
                parsed.get("relevant", False)
            )

            confidence = self.normalize_confidence(
                parsed.get("confidence", 0.0)
            )

            reason = str(
                parsed.get("reason", "")
            ).strip()

            accepted = (
                relevant
                and confidence >= self.confidence_threshold
            )

            return {
                "chunk_id": chunk_id,
                "relevant": relevant,
                "confidence": confidence,
                "accepted": accepted,
                "reason": reason,
                "status": "SUCCESS",
            }

        except Exception as exc:

            return {
                "chunk_id": chunk_id,
                "relevant": False,
                "confidence": 0.0,
                "accepted": False,
                "reason": f"Validation failed: {exc}",
                "status": "FAILED",
            }

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save_json(
        self,
        path: Path,
        data,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )

    # ---------------------------------------------------------
    # Run validation
    # ---------------------------------------------------------

    def run(self) -> List[Dict]:

        candidates = self.load_candidates()

        self.output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        validation_results: List[Dict] = []
        ground_truth: List[Dict] = []

        total_questions = len(candidates)

        total_candidates = 0
        total_relevant = 0
        total_failures = 0

        print("\n" + "=" * 80)
        print("LexMindAI Benchmark Validation")
        print("=" * 80)
        print(f"Questions            : {total_questions}")
        print(
            f"Confidence threshold : "
            f"{self.confidence_threshold:.2f}"
        )
        print("=" * 80)

        for question_number, sample in enumerate(
            candidates,
            start=1,
        ):

            question_id = sample.get(
                "question_id",
                sample.get("id", question_number),
            )

            question = sample.get(
                "question",
                "",
            ).strip()

            candidate_chunks = sample.get(
                "candidate_chunks",
                [],
            )

            if not question:
                print(
                    f"\n[{question_number}/{total_questions}] "
                    "Skipped: missing question"
                )
                continue

            print(
                f"\n[{question_number}/{total_questions}] "
                f"{question}"
            )

            relevant_chunk_ids: List[str] = []

            question_validations: List[Dict] = []

            for candidate_number, chunk in enumerate(
                candidate_chunks,
                start=1,
            ):

                total_candidates += 1

                verdict = self.validate_chunk(
                    question,
                    chunk,
                )

                if verdict["status"] == "FAILED":
                    total_failures += 1

                if verdict["accepted"]:
                    chunk_id = verdict["chunk_id"]

                    if (
                        chunk_id
                        and chunk_id not in relevant_chunk_ids
                    ):
                        relevant_chunk_ids.append(
                            chunk_id
                        )

                        total_relevant += 1

                record = {
                    "question_id": question_id,
                    "question": question,
                    "candidate_rank": candidate_number,
                    "chunk_id": verdict["chunk_id"],
                    "retrieval_score": chunk.get(
                        "score",
                        0.0,
                    ),
                    "source": chunk.get("source"),
                    "title": chunk.get("title"),
                    "section_heading": chunk.get(
                        "section_heading"
                    ),
                    "relevant": verdict["relevant"],
                    "confidence": verdict["confidence"],
                    "accepted": verdict["accepted"],
                    "reason": verdict["reason"],
                    "status": verdict["status"],
                }

                validation_results.append(record)
                question_validations.append(record)

                symbol = (
                    "✓"
                    if verdict["accepted"]
                    else "✗"
                )

                print(
                    f"    {candidate_number:02d}. "
                    f"{symbol} "
                    f"confidence="
                    f"{verdict['confidence']:.2f}"
                )

            ground_truth.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "relevant_chunk_ids": (
                        relevant_chunk_ids
                    ),
                    "relevant_count": len(
                        relevant_chunk_ids
                    ),
                }
            )

            print(
                f"    Relevant: "
                f"{len(relevant_chunk_ids)}/"
                f"{len(candidate_chunks)}"
            )

        validation_path = (
            self.output_path
            / "validation_results.json"
        )

        ground_truth_path = (
            self.output_path
            / "ground_truth.json"
        )

        self.save_json(
            validation_path,
            validation_results,
        )

        self.save_json(
            ground_truth_path,
            ground_truth,
        )

        print("\n" + "=" * 80)
        print("Benchmark Validation Complete")
        print("=" * 80)
        print(
            f"Questions processed : "
            f"{len(ground_truth)}"
        )
        print(
            f"Candidates judged   : "
            f"{total_candidates}"
        )
        print(
            f"Relevant accepted   : "
            f"{total_relevant}"
        )
        print(
            f"Validation failures : "
            f"{total_failures}"
        )

        if total_candidates:

            percentage = (
                total_relevant
                / total_candidates
                * 100
            )

            print(
                f"Relevant percentage : "
                f"{percentage:.2f}%"
            )

        print(
            f"Validation results  : "
            f"{validation_path}"
        )

        print(
            f"Ground truth        : "
            f"{ground_truth_path}"
        )

        print("=" * 80 + "\n")

        return ground_truth