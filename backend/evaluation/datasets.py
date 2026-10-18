"""
Benchmark Dataset Utilities

Provides utilities for loading and validating
evaluation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class BenchmarkDataset:
    """
    Loads benchmark datasets from JSON files.
    """

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

    def load(self) -> List[Dict]:
        """
        Load benchmark dataset.
        """

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        self.validate(data)

        return data

    @staticmethod
    def validate(data: List[Dict]) -> None:
        """
        Validate benchmark dataset structure.
        """

        if not isinstance(data, list):

            raise ValueError(
                "Benchmark dataset must be a list."
            )

        required = {
            "question",
            "relevant",
        }

        for index, sample in enumerate(data):

            missing = required - sample.keys()

            if missing:

                raise ValueError(
                    f"Sample {index} missing fields: {missing}"
                )

            if not isinstance(sample["relevant"], list):

                raise ValueError(
                    f"Sample {index}: 'relevant' must be a list."
                )

    @staticmethod
    def to_set(dataset: List[Dict]) -> List[Dict]:
        """
        Convert relevant lists to sets for faster evaluation.
        """

        converted = []

        for sample in dataset:

            converted.append(
                {
                    "question": sample["question"],
                    "relevant": set(sample["relevant"]),
                }
            )

        return converted