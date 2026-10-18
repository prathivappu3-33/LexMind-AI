"""
LexMindAI Human Ground Truth Reviewer

Interactive human verification tool for retrieval candidates.

Modes
-----
1. Continue review
2. Review/edit previous decisions by question
3. Review skipped candidates
4. Show summary
5. Exit

Controls during review
----------------------
y = Relevant
n = Not relevant
s = Skip / uncertain
k = Keep existing decision (edit mode only)
q = Return to main menu

All decisions are saved immediately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Paths
# ============================================================

CANDIDATE_PATH = Path(
    "datasets/ground_truth/candidate_ground_truth.json"
)

LLM_VALIDATION_PATH = Path(
    "datasets/ground_truth/validation_results.json"
)

PROGRESS_PATH = Path(
    "datasets/ground_truth/human_review_progress.json"
)

GROUND_TRUTH_PATH = Path(
    "datasets/ground_truth/ground_truth.json"
)


# ============================================================
# JSON Utilities
# ============================================================

def load_json(path: Path, required: bool = True):

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        return []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:

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


# ============================================================
# Lookup Builders
# ============================================================

def build_llm_lookup(
    validations: List[Dict],
) -> Dict:

    lookup = {}

    for item in validations:

        question_id = item.get("question_id")
        chunk_id = item.get("chunk_id")

        if question_id is None or not chunk_id:
            continue

        key = (
            str(question_id),
            str(chunk_id),
        )

        lookup[key] = item

    return lookup


def build_progress_lookup(
    progress: List[Dict],
) -> Dict:

    lookup = {}

    for item in progress:

        question_id = item.get("question_id")
        chunk_id = item.get("chunk_id")

        if question_id is None or not chunk_id:
            continue

        key = (
            str(question_id),
            str(chunk_id),
        )

        lookup[key] = item

    return lookup


# ============================================================
# Counting
# ============================================================

def count_total_candidates(
    candidates: List[Dict],
) -> int:

    return sum(
        len(sample.get("candidate_chunks", []))
        for sample in candidates
    )


def count_decisions(
    progress_lookup: Dict,
):

    relevant = 0
    not_relevant = 0
    skipped = 0

    for item in progress_lookup.values():

        decision = item.get("decision")

        if decision == "relevant":
            relevant += 1

        elif decision == "not_relevant":
            not_relevant += 1

        elif decision == "skipped":
            skipped += 1

    processed = (
        relevant
        + not_relevant
        + skipped
    )

    return (
        processed,
        relevant,
        not_relevant,
        skipped,
    )


# ============================================================
# Progress Saving
# ============================================================

def save_progress(
    progress_lookup: Dict,
) -> None:

    data = list(
        progress_lookup.values()
    )

    def sort_key(item):

        try:
            question_id = int(
                item.get("question_id", 0)
            )
        except (TypeError, ValueError):
            question_id = 0

        try:
            candidate_rank = int(
                item.get("candidate_rank", 0)
            )
        except (TypeError, ValueError):
            candidate_rank = 0

        return (
            question_id,
            candidate_rank,
        )

    data.sort(key=sort_key)

    save_json(
        PROGRESS_PATH,
        data,
    )


# ============================================================
# Ground Truth Generation
# ============================================================

def generate_ground_truth(
    candidates: List[Dict],
    progress_lookup: Dict,
) -> List[Dict]:

    ground_truth = []

    for question_number, sample in enumerate(
        candidates,
        start=1,
    ):

        question_id = sample.get(
            "question_id",
            sample.get(
                "id",
                question_number,
            ),
        )

        question = sample.get(
            "question",
            "",
        )

        relevant_chunk_ids = []
        skipped_chunk_ids = []

        reviewed_count = 0

        chunks = sample.get(
            "candidate_chunks",
            [],
        )

        for chunk in chunks:

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                continue

            key = (
                str(question_id),
                str(chunk_id),
            )

            review = progress_lookup.get(key)

            if not review:
                continue

            decision = review.get("decision")

            if decision in {
                "relevant",
                "not_relevant",
                "skipped",
            }:
                reviewed_count += 1

            if decision == "relevant":

                if (
                    chunk_id
                    not in relevant_chunk_ids
                ):
                    relevant_chunk_ids.append(
                        chunk_id
                    )

            elif decision == "skipped":

                if (
                    chunk_id
                    not in skipped_chunk_ids
                ):
                    skipped_chunk_ids.append(
                        chunk_id
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
                "skipped_chunk_ids": (
                    skipped_chunk_ids
                ),
                "reviewed_count": reviewed_count,
                "candidate_count": len(chunks),
                "validation_method": (
                    "human_verified"
                ),
            }
        )

    return ground_truth


def save_current_state(
    candidates: List[Dict],
    progress_lookup: Dict,
) -> None:

    save_progress(
        progress_lookup
    )

    ground_truth = generate_ground_truth(
        candidates,
        progress_lookup,
    )

    save_json(
        GROUND_TRUTH_PATH,
        ground_truth,
    )


# ============================================================
# Candidate Utilities
# ============================================================

def candidate_text(
    chunk: Dict,
) -> str:

    return str(
        chunk.get("text")
        or chunk.get("preview")
        or ""
    ).strip()


def decision_display(
    decision: Optional[str],
) -> str:

    if decision == "relevant":
        return "RELEVANT"

    if decision == "not_relevant":
        return "NOT RELEVANT"

    if decision == "skipped":
        return "SKIPPED / UNCERTAIN"

    return "NOT REVIEWED"


# ============================================================
# Candidate Display
# ============================================================

def print_candidate(
    question_number: int,
    total_questions: int,
    candidate_number: int,
    total_candidates_question: int,
    question: str,
    chunk: Dict,
    llm_result: Optional[Dict],
    existing_decision: Optional[str] = None,
    edit_mode: bool = False,
) -> None:

    print("\n" + "=" * 100)

    print(
        f"QUESTION "
        f"{question_number}/{total_questions}"
    )

    print("=" * 100)

    print(f"\n{question}\n")

    print("-" * 100)

    print(
        f"CANDIDATE "
        f"{candidate_number}/"
        f"{total_candidates_question}"
    )

    print("-" * 100)

    print(
        f"Chunk ID : "
        f"{chunk.get('chunk_id', 'N/A')}"
    )

    print(
        f"Title    : "
        f"{chunk.get('title') or 'N/A'}"
    )

    print(
        f"Section  : "
        f"{chunk.get('section_heading') or 'N/A'}"
    )

    score = chunk.get("score")

    if score is not None:

        try:

            print(
                f"Score    : "
                f"{float(score):.4f}"
            )

        except (TypeError, ValueError):

            print(
                f"Score    : {score}"
            )

    if edit_mode:

        print(
            f"Current decision : "
            f"{decision_display(existing_decision)}"
        )

    print("\nPASSAGE\n")

    text = candidate_text(chunk)

    if text:
        print(text)

    else:
        print(
            "[No passage text available]"
        )

    print("\n" + "-" * 100)

    if llm_result:

        accepted = llm_result.get(
            "accepted",
            False,
        )

        relevant = llm_result.get(
            "relevant",
            False,
        )

        confidence = llm_result.get(
            "confidence",
            0.0,
        )

        reason = llm_result.get(
            "reason",
            "",
        )

        print(
            "Gemma suggestion : "
            + (
                "RELEVANT"
                if accepted
                else "NOT RELEVANT"
            )
        )

        print(
            f"Gemma raw label  : "
            f"{relevant}"
        )

        try:

            print(
                f"Gemma confidence : "
                f"{float(confidence):.2f}"
            )

        except (TypeError, ValueError):

            print(
                f"Gemma confidence : "
                f"{confidence}"
            )

        if reason:

            print(
                f"Gemma reason     : "
                f"{reason}"
            )

    else:

        print(
            "Gemma suggestion : unavailable"
        )

    print("-" * 100)

    print("\nLABELING RULE")

    print(
        "y = The passage contains useful evidence "
        "that can contribute to answering the question."
    )

    print(
        "n = The passage does not provide useful "
        "evidence for this specific question."
    )

    print(
        "s = You genuinely cannot decide."
    )

    print(
        "\nA passage does NOT need to answer the "
        "entire question by itself to be relevant."
    )

    print(
        "Sharing Docker-related keywords alone "
        "is NOT enough."
    )

    if edit_mode:

        print(
            "\n[y] Relevant"
            "   [n] Not relevant"
            "   [s] Skip/uncertain"
            "   [k] Keep current"
            "   [q] Main menu"
        )

    else:

        print(
            "\n[y] Relevant"
            "   [n] Not relevant"
            "   [s] Skip/uncertain"
            "   [q] Main menu"
        )


# ============================================================
# Summary
# ============================================================

def print_summary(
    candidates: List[Dict],
    progress_lookup: Dict,
) -> None:

    total = count_total_candidates(
        candidates
    )

    (
        processed,
        relevant,
        not_relevant,
        skipped,
    ) = count_decisions(
        progress_lookup
    )

    remaining = max(
        total - processed,
        0,
    )

    print("\n" + "=" * 100)

    print("HUMAN REVIEW SUMMARY")

    print("=" * 100)

    print(
        f"Total candidates : {total}"
    )

    print(
        f"Processed        : {processed}"
    )

    print(
        f"Relevant         : {relevant}"
    )

    print(
        f"Not relevant     : {not_relevant}"
    )

    print(
        f"Skipped          : {skipped}"
    )

    print(
        f"Remaining        : {remaining}"
    )

    if total:

        percentage = (
            processed
            / total
            * 100
        )

        print(
            f"Progress         : "
            f"{percentage:.2f}%"
        )

    print("=" * 100)


# ============================================================
# Save Decision
# ============================================================

def store_decision(
    progress_lookup: Dict,
    candidates: List[Dict],
    question_id,
    question: str,
    candidate_number: int,
    chunk: Dict,
    decision: str,
) -> None:

    chunk_id = chunk.get("chunk_id")

    key = (
        str(question_id),
        str(chunk_id),
    )

    progress_lookup[key] = {
        "question_id": question_id,
        "question": question,
        "candidate_rank": candidate_number,
        "chunk_id": chunk_id,
        "decision": decision,
        "title": chunk.get("title"),
        "section_heading": (
            chunk.get("section_heading")
        ),
    }

    save_current_state(
        candidates,
        progress_lookup,
    )


# ============================================================
# Mode 1 - Continue Review
# ============================================================

def continue_review(
    candidates: List[Dict],
    llm_lookup: Dict,
    progress_lookup: Dict,
) -> None:

    total_questions = len(candidates)

    total_global = count_total_candidates(
        candidates
    )

    for question_number, sample in enumerate(
        candidates,
        start=1,
    ):

        question_id = sample.get(
            "question_id",
            sample.get(
                "id",
                question_number,
            ),
        )

        question = sample.get(
            "question",
            "",
        )

        chunks = sample.get(
            "candidate_chunks",
            [],
        )

        for candidate_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                continue

            key = (
                str(question_id),
                str(chunk_id),
            )

            existing = progress_lookup.get(key)

            if existing:

                decision = existing.get(
                    "decision"
                )

                if decision in {
                    "relevant",
                    "not_relevant",
                    "skipped",
                }:
                    continue

            llm_result = llm_lookup.get(key)

            print_candidate(
                question_number,
                total_questions,
                candidate_number,
                len(chunks),
                question,
                chunk,
                llm_result,
                edit_mode=False,
            )

            while True:

                choice = input(
                    "\nYour decision: "
                ).strip().lower()

                if choice in {
                    "y",
                    "n",
                    "s",
                    "q",
                }:
                    break

                print(
                    "Enter y, n, s, or q."
                )

            if choice == "q":

                save_current_state(
                    candidates,
                    progress_lookup,
                )

                return

            if choice == "y":
                decision = "relevant"

            elif choice == "n":
                decision = "not_relevant"

            else:
                decision = "skipped"

            store_decision(
                progress_lookup,
                candidates,
                question_id,
                question,
                candidate_number,
                chunk,
                decision,
            )

            (
                processed,
                relevant,
                not_relevant,
                skipped,
            ) = count_decisions(
                progress_lookup
            )

            print(
                f"\nSaved ✓"
                f" | Processed: "
                f"{processed}/{total_global}"
                f" | Relevant: {relevant}"
                f" | Not relevant: "
                f"{not_relevant}"
                f" | Skipped: {skipped}"
            )

    print("\nAll candidates have been processed.")

    print_summary(
        candidates,
        progress_lookup,
    )


# ============================================================
# Mode 2 - Edit Question
# ============================================================

def edit_question(
    candidates: List[Dict],
    llm_lookup: Dict,
    progress_lookup: Dict,
) -> None:

    print("\nAVAILABLE QUESTIONS\n")

    for index, sample in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"{index:>2}. "
            f"{sample.get('question', '')}"
        )

    while True:

        raw = input(
            "\nQuestion number "
            "(or q for menu): "
        ).strip().lower()

        if raw == "q":
            return

        try:
            question_number = int(raw)

        except ValueError:

            print(
                "Enter a valid question number."
            )

            continue

        if not (
            1
            <= question_number
            <= len(candidates)
        ):

            print(
                "Question number out of range."
            )

            continue

        break

    sample = candidates[
        question_number - 1
    ]

    question_id = sample.get(
        "question_id",
        sample.get(
            "id",
            question_number,
        ),
    )

    question = sample.get(
        "question",
        "",
    )

    chunks = sample.get(
        "candidate_chunks",
        [],
    )

    for candidate_number, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            continue

        key = (
            str(question_id),
            str(chunk_id),
        )

        existing = progress_lookup.get(
            key
        )

        existing_decision = None

        if existing:
            existing_decision = (
                existing.get("decision")
            )

        llm_result = llm_lookup.get(key)

        print_candidate(
            question_number,
            len(candidates),
            candidate_number,
            len(chunks),
            question,
            chunk,
            llm_result,
            existing_decision=(
                existing_decision
            ),
            edit_mode=True,
        )

        while True:

            choice = input(
                "\nYour decision: "
            ).strip().lower()

            if choice in {
                "y",
                "n",
                "s",
                "k",
                "q",
            }:
                break

            print(
                "Enter y, n, s, k, or q."
            )

        if choice == "q":

            save_current_state(
                candidates,
                progress_lookup,
            )

            return

        if choice == "k":

            continue

        if choice == "y":
            decision = "relevant"

        elif choice == "n":
            decision = "not_relevant"

        else:
            decision = "skipped"

        store_decision(
            progress_lookup,
            candidates,
            question_id,
            question,
            candidate_number,
            chunk,
            decision,
        )

        print(
            f"\nUpdated ✓ "
            f"{decision_display(decision)}"
        )

    print(
        f"\nQuestion {question_number} "
        f"review complete."
    )


# ============================================================
# Mode 3 - Review Skipped
# ============================================================

def review_skipped(
    candidates: List[Dict],
    llm_lookup: Dict,
    progress_lookup: Dict,
) -> None:

    skipped_items = []

    for question_number, sample in enumerate(
        candidates,
        start=1,
    ):

        question_id = sample.get(
            "question_id",
            sample.get(
                "id",
                question_number,
            ),
        )

        question = sample.get(
            "question",
            "",
        )

        chunks = sample.get(
            "candidate_chunks",
            [],
        )

        for candidate_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                continue

            key = (
                str(question_id),
                str(chunk_id),
            )

            review = progress_lookup.get(key)

            if (
                review
                and review.get("decision")
                == "skipped"
            ):

                skipped_items.append(
                    (
                        question_number,
                        question_id,
                        question,
                        candidate_number,
                        len(chunks),
                        chunk,
                        key,
                    )
                )

    if not skipped_items:

        print(
            "\nThere are no skipped "
            "candidates to review."
        )

        return

    print(
        f"\nSkipped candidates: "
        f"{len(skipped_items)}"
    )

    for (
        question_number,
        question_id,
        question,
        candidate_number,
        question_candidate_count,
        chunk,
        key,
    ) in skipped_items:

        llm_result = llm_lookup.get(key)

        print_candidate(
            question_number,
            len(candidates),
            candidate_number,
            question_candidate_count,
            question,
            chunk,
            llm_result,
            existing_decision="skipped",
            edit_mode=True,
        )

        while True:

            choice = input(
                "\nYour decision: "
            ).strip().lower()

            if choice in {
                "y",
                "n",
                "s",
                "k",
                "q",
            }:
                break

            print(
                "Enter y, n, s, k, or q."
            )

        if choice == "q":

            save_current_state(
                candidates,
                progress_lookup,
            )

            return

        if choice in {
            "s",
            "k",
        }:

            continue

        if choice == "y":
            decision = "relevant"

        else:
            decision = "not_relevant"

        store_decision(
            progress_lookup,
            candidates,
            question_id,
            question,
            candidate_number,
            chunk,
            decision,
        )

        print(
            f"\nUpdated ✓ "
            f"{decision_display(decision)}"
        )

    print(
        "\nSkipped-candidate review complete."
    )


# ============================================================
# Main Menu
# ============================================================

def main() -> None:

    candidates = load_json(
        CANDIDATE_PATH
    )

    validations = load_json(
        LLM_VALIDATION_PATH,
        required=False,
    )

    previous_progress = load_json(
        PROGRESS_PATH,
        required=False,
    )

    llm_lookup = build_llm_lookup(
        validations
    )

    progress_lookup = (
        build_progress_lookup(
            previous_progress
        )
    )

    # Always synchronize ground_truth.json
    # with current human review progress.
    save_current_state(
        candidates,
        progress_lookup,
    )

    print("\n" + "=" * 100)

    print(
        "LexMindAI Human Ground Truth Review"
    )

    print("=" * 100)

    if previous_progress:

        print(
            f"Loaded "
            f"{len(progress_lookup)} "
            f"saved decisions."
        )

    else:

        print(
            "No previous human review found."
        )

    print_summary(
        candidates,
        progress_lookup,
    )

    while True:

        print(
            "\nMAIN MENU"
        )

        print(
            "1. Continue review"
        )

        print(
            "2. Review/edit previous decisions"
        )

        print(
            "3. Review skipped candidates"
        )

        print(
            "4. Show summary"
        )

        print(
            "5. Exit"
        )

        choice = input(
            "\nChoose option: "
        ).strip()

        if choice == "1":

            continue_review(
                candidates,
                llm_lookup,
                progress_lookup,
            )

        elif choice == "2":

            edit_question(
                candidates,
                llm_lookup,
                progress_lookup,
            )

        elif choice == "3":

            review_skipped(
                candidates,
                llm_lookup,
                progress_lookup,
            )

        elif choice == "4":

            print_summary(
                candidates,
                progress_lookup,
            )

        elif choice == "5":

            save_current_state(
                candidates,
                progress_lookup,
            )

            print(
                "\nProgress saved."
            )

            print(
                f"Human review : "
                f"{PROGRESS_PATH}"
            )

            print(
                f"Ground truth : "
                f"{GROUND_TRUTH_PATH}"
            )

            return

        else:

            print(
                "\nInvalid option. "
                "Choose 1, 2, 3, 4, or 5."
            )


if __name__ == "__main__":
    main()