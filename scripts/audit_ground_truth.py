"""
LexMindAI Ground Truth Consistency Audit

Audits the human-reviewed retrieval ground truth before benchmarking.

The script DOES NOT modify any labels.

It checks:
    1. Human vs Gemma validator disagreements
    2. Suspicious high-score NOT RELEVANT labels
    3. Suspicious low-score RELEVANT labels
    4. Missing / malformed fields
    5. Duplicate question-chunk pairs
    6. Questions with no relevant chunks
    7. Questions with unusually many/few relevant chunks

Outputs:
    datasets/ground_truth/audit_report.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

CANDIDATE_PATH = Path(
    "datasets/ground_truth/candidate_ground_truth.json"
)

HUMAN_REVIEW_PATH = Path(
    "datasets/ground_truth/human_review_progress.json"
)

VALIDATION_PATH = Path(
    "datasets/ground_truth/validation_results.json"
)

GROUND_TRUTH_PATH = Path(
    "datasets/ground_truth/ground_truth.json"
)

OUTPUT_PATH = Path(
    "datasets/ground_truth/audit_report.json"
)


# Retrieval-score thresholds used only for FLAGGING.
# They do NOT determine relevance automatically.

HIGH_SCORE_THRESHOLD = 0.78
LOW_SCORE_THRESHOLD = 0.70

# Gemma confidence required before disagreement is highlighted.
GEMMA_CONFIDENCE_THRESHOLD = 0.90


# =============================================================================
# Utility Functions
# =============================================================================

def load_json(path: Path) -> Any:
    """
    Load a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalize_decision(value: Any) -> Optional[bool]:
    """
    Convert different human-review decision formats into:

        True  -> relevant
        False -> not relevant
        None  -> skipped / unknown
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return None

    if isinstance(value, str):

        normalized = value.strip().lower()

        relevant_values = {
            "y",
            "yes",
            "true",
            "relevant"
        }

        not_relevant_values = {
            "n",
            "no",
            "false",
            "not relevant",
            "not_relevant",
            "irrelevant"
        }

        skipped_values = {
            "s",
            "skip",
            "skipped",
            "uncertain",
            "unknown",
            ""
        }

        if normalized in relevant_values:
            return True

        if normalized in not_relevant_values:
            return False

        if normalized in skipped_values:
            return None

    return None


def safe_float(value: Any) -> float:
    """
    Convert a value to float safely.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return 0.0


def truncate_text(
    text: Any,
    limit: int = 300
) -> str:
    """
    Return a short one-line preview.
    """

    if text is None:
        return ""

    text = str(text)

    text = " ".join(
        text.split()
    )

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


# =============================================================================
# Candidate Index
# =============================================================================

def build_candidate_index(
    candidate_data: List[Dict]
) -> Dict[Tuple[str, str], Dict]:
    """
    Build:

        (question_id, chunk_id) -> candidate
    """

    index: Dict[
        Tuple[str, str],
        Dict
    ] = {}

    for question_entry in candidate_data:

        question_id = str(
            question_entry.get(
                "question_id",
                ""
            )
        )

        question = question_entry.get(
            "question",
            ""
        )

        candidates = question_entry.get(
            "candidate_chunks",
            []
        )

        for candidate in candidates:

            chunk_id = candidate.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            key = (
                question_id,
                str(chunk_id)
            )

            record = dict(candidate)

            record["question_id"] = question_id
            record["question"] = question

            index[key] = record

    return index


# =============================================================================
# Validation Index
# =============================================================================

def build_validation_index(
    validation_data: List[Dict]
) -> Dict[Tuple[str, str], Dict]:
    """
    Build:

        (question_id, chunk_id) -> validator result
    """

    index: Dict[
        Tuple[str, str],
        Dict
    ] = {}

    for item in validation_data:

        question_id = str(
            item.get(
                "question_id",
                ""
            )
        )

        chunk_id = item.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        key = (
            question_id,
            str(chunk_id)
        )

        index[key] = item

    return index


# =============================================================================
# Human Review Extraction
# =============================================================================

def extract_human_decisions(
    human_data: Any
) -> List[Dict]:
    """
    Convert human_review_progress.json into a normalized list.

    Supports several likely formats so the audit remains robust.
    """

    normalized: List[Dict] = []

    # -------------------------------------------------------------------------
    # Format 1:
    #
    # [
    #   {
    #       "question_id": 1,
    #       "chunk_id": "...",
    #       "decision": "y"
    #   }
    # ]
    # -------------------------------------------------------------------------

    if isinstance(human_data, list):

        for item in human_data:

            if not isinstance(
                item,
                dict
            ):
                continue

            question_id = item.get(
                "question_id"
            )

            chunk_id = item.get(
                "chunk_id"
            )

            decision = item.get(
                "decision"
            )

            if decision is None:
                decision = item.get(
                    "human_decision"
                )

            if decision is None:
                decision = item.get(
                    "relevant"
                )

            normalized.append(
                {
                    "question_id": str(
                        question_id
                    ),
                    "chunk_id": chunk_id,
                    "decision": normalize_decision(
                        decision
                    ),
                    "raw": item
                }
            )

        return normalized

    # -------------------------------------------------------------------------
    # Dictionary formats
    # -------------------------------------------------------------------------

    if isinstance(human_data, dict):

        # Possible wrapper:
        #
        # {
        #     "decisions": [...]
        # }

        for wrapper_key in (
            "decisions",
            "reviews",
            "progress",
            "items"
        ):

            wrapped = human_data.get(
                wrapper_key
            )

            if isinstance(
                wrapped,
                list
            ):
                return extract_human_decisions(
                    wrapped
                )

        # Possible mapping:
        #
        # {
        #   "1:chunk-id": "y",
        #   "1:chunk-id-2": "n"
        # }
        #
        # or
        #
        # {
        #   "1": {
        #       "chunk-id": "y"
        #   }
        # }

        for key, value in human_data.items():

            # Nested question -> chunks mapping

            if isinstance(
                value,
                dict
            ):

                # Could itself be a decision record

                if (
                    "chunk_id" in value
                    or
                    "decision" in value
                    or
                    "human_decision" in value
                    or
                    "relevant" in value
                ):

                    question_id = value.get(
                        "question_id",
                        key
                    )

                    chunk_id = value.get(
                        "chunk_id"
                    )

                    decision = value.get(
                        "decision"
                    )

                    if decision is None:
                        decision = value.get(
                            "human_decision"
                        )

                    if decision is None:
                        decision = value.get(
                            "relevant"
                        )

                    normalized.append(
                        {
                            "question_id": str(
                                question_id
                            ),
                            "chunk_id": chunk_id,
                            "decision": normalize_decision(
                                decision
                            ),
                            "raw": value
                        }
                    )

                    continue

                # Assume:
                #
                # question_id -> chunk_id -> decision

                question_id = str(
                    key
                )

                for chunk_id, decision in value.items():

                    normalized.append(
                        {
                            "question_id": question_id,
                            "chunk_id": chunk_id,
                            "decision": normalize_decision(
                                decision
                            ),
                            "raw": decision
                        }
                    )

                continue

            # Flat compound key

            compound_key = str(
                key
            )

            separators = [
                "::",
                "|",
                ":"
            ]

            parsed = False

            for separator in separators:

                if separator in compound_key:

                    question_id, chunk_id = (
                        compound_key.split(
                            separator,
                            1
                        )
                    )

                    normalized.append(
                        {
                            "question_id": str(
                                question_id
                            ),
                            "chunk_id": chunk_id,
                            "decision": normalize_decision(
                                value
                            ),
                            "raw": value
                        }
                    )

                    parsed = True
                    break

            if parsed:
                continue

    return normalized


# =============================================================================
# Ground Truth Checks
# =============================================================================

def audit_ground_truth_file(
    ground_truth_data: List[Dict]
) -> Dict:
    """
    Inspect the finalized ground_truth.json structure.
    """

    issues: List[Dict] = []

    question_ids = []

    total_relevant_ids = 0

    relevant_counts = {}

    for entry in ground_truth_data:

        question_id = entry.get(
            "question_id"
        )

        question = entry.get(
            "question"
        )

        relevant_ids = entry.get(
            "relevant_chunk_ids"
        )

        question_ids.append(
            question_id
        )

        if not question:
            issues.append(
                {
                    "type": "MISSING_QUESTION",
                    "question_id": question_id
                }
            )

        if not isinstance(
            relevant_ids,
            list
        ):

            issues.append(
                {
                    "type": "INVALID_RELEVANT_CHUNK_IDS",
                    "question_id": question_id
                }
            )

            continue

        total_relevant_ids += len(
            relevant_ids
        )

        relevant_counts[
            str(question_id)
        ] = len(
            relevant_ids
        )

        if len(relevant_ids) == 0:

            issues.append(
                {
                    "type": "NO_RELEVANT_CHUNKS",
                    "question_id": question_id,
                    "question": question
                }
            )

        duplicates = [
            chunk_id
            for chunk_id, count
            in Counter(
                relevant_ids
            ).items()
            if count > 1
        ]

        if duplicates:

            issues.append(
                {
                    "type": "DUPLICATE_RELEVANT_CHUNKS",
                    "question_id": question_id,
                    "chunk_ids": duplicates
                }
            )

    duplicate_question_ids = [
        question_id
        for question_id, count
        in Counter(
            question_ids
        ).items()
        if count > 1
    ]

    if duplicate_question_ids:

        issues.append(
            {
                "type": "DUPLICATE_QUESTION_IDS",
                "question_ids": duplicate_question_ids
            }
        )

    return {
        "questions": len(
            ground_truth_data
        ),
        "total_relevant_chunk_ids": total_relevant_ids,
        "relevant_counts": relevant_counts,
        "issues": issues
    }


# =============================================================================
# Main Audit
# =============================================================================

def run_audit() -> Dict:
    """
    Execute the complete consistency audit.
    """

    print(
        "\n" +
        "=" * 100
    )

    print(
        "LexMindAI Ground Truth Consistency Audit"
    )

    print(
        "=" * 100
    )

    print(
        "\nLoading benchmark files...\n"
    )

    candidate_data = load_json(
        CANDIDATE_PATH
    )

    human_data = load_json(
        HUMAN_REVIEW_PATH
    )

    validation_data = load_json(
        VALIDATION_PATH
    )

    ground_truth_data = load_json(
        GROUND_TRUTH_PATH
    )

    candidate_index = build_candidate_index(
        candidate_data
    )

    validation_index = build_validation_index(
        validation_data
    )

    human_decisions = extract_human_decisions(
        human_data
    )

    print(
        f"Candidate pairs       : {len(candidate_index)}"
    )

    print(
        f"Human decisions       : {len(human_decisions)}"
    )

    print(
        f"Validator results     : {len(validation_index)}"
    )

    print(
        f"Ground truth questions: {len(ground_truth_data)}"
    )

    # =========================================================================
    # Audit Containers
    # =========================================================================

    flags: List[Dict] = []

    missing_candidates: List[Dict] = []

    duplicate_pairs: List[Dict] = []

    seen_pairs = set()

    relevant_count = 0
    not_relevant_count = 0
    unknown_count = 0

    question_stats = defaultdict(
        lambda: {
            "relevant": 0,
            "not_relevant": 0,
            "unknown": 0,
            "total": 0
        }
    )

    # =========================================================================
    # Human Decision Audit
    # =========================================================================

    for human_item in human_decisions:

        question_id = str(
            human_item.get(
                "question_id",
                ""
            )
        )

        chunk_id = human_item.get(
            "chunk_id"
        )

        decision = human_item.get(
            "decision"
        )

        if not chunk_id:

            flags.append(
                {
                    "flag_type": "MISSING_CHUNK_ID",
                    "question_id": question_id
                }
            )

            continue

        pair = (
            question_id,
            str(chunk_id)
        )

        if pair in seen_pairs:

            duplicate_pairs.append(
                {
                    "question_id": question_id,
                    "chunk_id": chunk_id
                }
            )

        seen_pairs.add(
            pair
        )

        candidate = candidate_index.get(
            pair
        )

        if candidate is None:

            missing_candidates.append(
                {
                    "question_id": question_id,
                    "chunk_id": chunk_id
                }
            )

            continue

        question = candidate.get(
            "question",
            ""
        )

        score = safe_float(
            candidate.get(
                "score"
            )
        )

        question_stats[
            question_id
        ]["total"] += 1

        if decision is True:

            relevant_count += 1

            question_stats[
                question_id
            ]["relevant"] += 1

        elif decision is False:

            not_relevant_count += 1

            question_stats[
                question_id
            ]["not_relevant"] += 1

        else:

            unknown_count += 1

            question_stats[
                question_id
            ]["unknown"] += 1

        validator = validation_index.get(
            pair
        )

        validator_relevant = None
        validator_confidence = 0.0
        validator_reason = ""

        if validator:

            validator_relevant = normalize_decision(
                validator.get(
                    "relevant"
                )
            )

            validator_confidence = safe_float(
                validator.get(
                    "confidence"
                )
            )

            validator_reason = validator.get(
                "reason",
                ""
            )

        # =====================================================================
        # Flag 1:
        # Human says NOT relevant despite high retrieval score
        # =====================================================================

        if (
            decision is False
            and
            score >= HIGH_SCORE_THRESHOLD
        ):

            flags.append(
                {
                    "flag_type":
                        "HIGH_SCORE_NOT_RELEVANT",

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "chunk_id":
                        chunk_id,

                    "retrieval_score":
                        score,

                    "human_label":
                        "NOT_RELEVANT",

                    "title":
                        candidate.get(
                            "title"
                        ),

                    "section":
                        candidate.get(
                            "section_heading"
                        ),

                    "passage":
                        truncate_text(
                            candidate.get(
                                "preview",
                                candidate.get(
                                    "text",
                                    ""
                                )
                            )
                        )
                }
            )

        # =====================================================================
        # Flag 2:
        # Human says relevant despite relatively low retrieval score
        #
        # This is NOT necessarily wrong.
        # Retrieval score and relevance are different concepts.
        # It is only worth inspecting.
        # =====================================================================

        if (
            decision is True
            and
            score < LOW_SCORE_THRESHOLD
        ):

            flags.append(
                {
                    "flag_type":
                        "LOW_SCORE_RELEVANT",

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "chunk_id":
                        chunk_id,

                    "retrieval_score":
                        score,

                    "human_label":
                        "RELEVANT",

                    "title":
                        candidate.get(
                            "title"
                        ),

                    "section":
                        candidate.get(
                            "section_heading"
                        ),

                    "passage":
                        truncate_text(
                            candidate.get(
                                "preview",
                                candidate.get(
                                    "text",
                                    ""
                                )
                            )
                        )
                }
            )

        # =====================================================================
        # Flag 3:
        # Human and high-confidence Gemma disagree
        # =====================================================================

        if (
            decision is not None
            and
            validator_relevant is not None
            and
            decision != validator_relevant
            and
            validator_confidence
            >= GEMMA_CONFIDENCE_THRESHOLD
        ):

            flags.append(
                {
                    "flag_type":
                        "HUMAN_GEMMA_DISAGREEMENT",

                    "question_id":
                        question_id,

                    "question":
                        question,

                    "chunk_id":
                        chunk_id,

                    "retrieval_score":
                        score,

                    "human_label":
                        (
                            "RELEVANT"
                            if decision
                            else "NOT_RELEVANT"
                        ),

                    "gemma_label":
                        (
                            "RELEVANT"
                            if validator_relevant
                            else "NOT_RELEVANT"
                        ),

                    "gemma_confidence":
                        validator_confidence,

                    "gemma_reason":
                        validator_reason,

                    "title":
                        candidate.get(
                            "title"
                        ),

                    "section":
                        candidate.get(
                            "section_heading"
                        ),

                    "passage":
                        truncate_text(
                            candidate.get(
                                "preview",
                                candidate.get(
                                    "text",
                                    ""
                                )
                            )
                        )
                }
            )

    # =========================================================================
    # Question-Level Checks
    # =========================================================================

    for question_id, stats in question_stats.items():

        if stats["total"] == 0:
            continue

        if stats["relevant"] == 0:

            candidate_question = ""

            for (
                candidate_question_id,
                _
            ), candidate in candidate_index.items():

                if (
                    candidate_question_id
                    == question_id
                ):

                    candidate_question = (
                        candidate.get(
                            "question",
                            ""
                        )
                    )

                    break

            flags.append(
                {
                    "flag_type":
                        "QUESTION_HAS_ZERO_RELEVANT",

                    "question_id":
                        question_id,

                    "question":
                        candidate_question,

                    "statistics":
                        stats
                }
            )

        relevance_ratio = (
            stats["relevant"]
            /
            stats["total"]
        )

        # This does not mean the labels are wrong.
        # It merely highlights questions where nearly every retrieved
        # candidate was marked relevant.

        if (
            stats["total"] >= 5
            and
            relevance_ratio >= 0.90
        ):

            flags.append(
                {
                    "flag_type":
                        "VERY_HIGH_RELEVANCE_RATIO",

                    "question_id":
                        question_id,

                    "relevance_ratio":
                        round(
                            relevance_ratio,
                            4
                        ),

                    "statistics":
                        stats
                }
            )

    # =========================================================================
    # Ground Truth File Audit
    # =========================================================================

    ground_truth_audit = audit_ground_truth_file(
        ground_truth_data
    )

    # =========================================================================
    # Build Report
    # =========================================================================

    flag_counts = Counter(
        item["flag_type"]
        for item in flags
    )

    report = {

        "summary": {

            "candidate_pairs":
                len(candidate_index),

            "human_decisions":
                len(human_decisions),

            "human_relevant":
                relevant_count,

            "human_not_relevant":
                not_relevant_count,

            "human_unknown":
                unknown_count,

            "validator_results":
                len(validation_index),

            "flags":
                len(flags),

            "missing_candidates":
                len(missing_candidates),

            "duplicate_human_pairs":
                len(duplicate_pairs)
        },

        "flag_counts":
            dict(flag_counts),

        "question_statistics":
            dict(question_stats),

        "flags":
            flags,

        "missing_candidates":
            missing_candidates,

        "duplicate_human_pairs":
            duplicate_pairs,

        "ground_truth_file_audit":
            ground_truth_audit
    }

    # =========================================================================
    # Save
    # =========================================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # =========================================================================
    # Terminal Summary
    # =========================================================================

    print(
        "\n" +
        "=" * 100
    )

    print(
        "HUMAN LABEL SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Relevant      : {relevant_count}"
    )

    print(
        f"Not relevant  : {not_relevant_count}"
    )

    print(
        f"Unknown       : {unknown_count}"
    )

    print(
        f"Total         : "
        f"{relevant_count + not_relevant_count + unknown_count}"
    )

    print(
        "\n" +
        "=" * 100
    )

    print(
        "AUDIT FLAGS"
    )

    print(
        "=" * 100
    )

    if not flag_counts:

        print(
            "No suspicious cases detected."
        )

    else:

        for flag_type, count in sorted(
            flag_counts.items()
        ):

            print(
                f"{flag_type:<35} : {count}"
            )

    print(
        "\n" +
        "=" * 100
    )

    print(
        "STRUCTURAL CHECKS"
    )

    print(
        "=" * 100
    )

    print(
        f"Missing candidate pairs : "
        f"{len(missing_candidates)}"
    )

    print(
        f"Duplicate human pairs   : "
        f"{len(duplicate_pairs)}"
    )

    print(
        f"Ground-truth issues     : "
        f"{len(ground_truth_audit['issues'])}"
    )

    print(
        "\n" +
        "=" * 100
    )

    print(
        "QUESTION RELEVANCE COUNTS"
    )

    print(
        "=" * 100
    )

    def question_sort_key(
        item: Tuple[str, Dict]
    ) -> Tuple[int, str]:

        question_id = item[0]

        try:
            return (
                int(question_id),
                question_id
            )

        except ValueError:
            return (
                10**9,
                question_id
            )

    for question_id, stats in sorted(
        question_stats.items(),
        key=question_sort_key
    ):

        print(
            f"Q{question_id:<3} | "
            f"Relevant: {stats['relevant']:>2} | "
            f"Not relevant: {stats['not_relevant']:>2} | "
            f"Unknown: {stats['unknown']:>2} | "
            f"Total: {stats['total']:>2}"
        )

    print(
        "\n" +
        "=" * 100
    )

    print(
        "Audit complete."
    )

    print(
        f"Full report: {OUTPUT_PATH}"
    )

    print(
        "=" * 100 +
        "\n"
    )

    return report


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()