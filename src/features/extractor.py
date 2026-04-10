"""Structural feature extraction using dual-extractor approach.

Extracts 5 interpretable structural features from task descriptions:
1. Relational density
2. Coordinate precision requirement
3. Hierarchical depth
4. Primitive count
5. Data-bound axis count

Uses two independent LLM extractors for automated reliability validation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TaskFeatures:
    """Extracted structural features for a task."""
    task_id: str
    relational_density: float  # entity-entity connections / total entities
    coordinate_precision: float  # fraction of tokens referencing positions
    hierarchical_depth: int  # max nesting depth
    primitive_count: int  # number of visual elements
    data_bound_axes: int  # 0 for diagrams, 1-3 for charts
    extractor_id: str  # "A" or "B"
    raw_response: dict[str, Any]

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.relational_density,
            self.coordinate_precision,
            self.hierarchical_depth,
            self.primitive_count,
            self.data_bound_axes,
        ], dtype=np.float64)


class DualFeatureExtractor:
    """Dual-extractor system for reliable feature extraction.

    Runs two independent LLM extractors with different prompt variants
    on every task, then computes agreement metrics to validate
    extraction reliability.

    Args:
        model_a: Model for extractor A (e.g., Qwen2.5-7B-Instruct).
        model_b: Model for extractor B (e.g., Qwen2.5-Coder-7B-Instruct).
        device: Device for inference.
    """

    def __init__(
        self,
        model_a: str = "Qwen/Qwen2.5-7B-Instruct",
        model_b: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "cuda",
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.device = device

    def _build_extraction_prompt(
        self, description: str, variant: str
    ) -> str:
        """Build JSON-schema extraction prompt.

        Args:
            description: Task description.
            variant: "A" or "B" for different prompt wording.

        Returns:
            Extraction prompt string.
        """
        raise NotImplementedError

    def extract_single(
        self, description: str, task_id: str, extractor: str = "A"
    ) -> TaskFeatures:
        """Extract features using one extractor.

        Args:
            description: Task description.
            task_id: Unique task identifier.
            extractor: "A" or "B".

        Returns:
            TaskFeatures instance.
        """
        raise NotImplementedError

    def extract_dual(
        self, description: str, task_id: str
    ) -> tuple[TaskFeatures, TaskFeatures]:
        """Extract features using both extractors.

        Args:
            description: Task description.
            task_id: Unique task identifier.

        Returns:
            Tuple of (features_A, features_B).
        """
        raise NotImplementedError

    def extract_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[TaskFeatures, TaskFeatures]]:
        """Extract dual features for a batch of (description, task_id) pairs.

        Args:
            items: List of (description, task_id).

        Returns:
            List of (features_A, features_B) tuples.
        """
        raise NotImplementedError

    def compute_agreement(
        self,
        features_a: list[TaskFeatures],
        features_b: list[TaskFeatures],
    ) -> dict[str, dict[str, float]]:
        """Compute per-feature agreement between extractors.

        Continuous features: Pearson r (threshold ≥ 0.75).
        Discrete features: Cohen's kappa (threshold ≥ 0.6).

        Args:
            features_a: Features from extractor A.
            features_b: Features from extractor B.

        Returns:
            Dict mapping feature_name -> {"metric": ..., "value": ..., "pass": bool}.
        """
        raise NotImplementedError

    def merge_features(
        self,
        features_a: list[TaskFeatures],
        features_b: list[TaskFeatures],
    ) -> list[TaskFeatures]:
        """Merge dual-extracted features by averaging (continuous) or majority (discrete).

        Args:
            features_a: Features from extractor A.
            features_b: Features from extractor B.

        Returns:
            Merged feature list.
        """
        raise NotImplementedError
