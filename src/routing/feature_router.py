"""VisRoute-Feature: interpretable format routing via structural features.

Routes tasks to the format with highest predicted execution success
probability, using the Part 2 logistic regression affinity model.
"""

from typing import Any

import numpy as np

from src.features.affinity_model import AffinityModel
from src.features.extractor import TaskFeatures


class FeatureRouter:
    """Interpretable format router based on structural task features.

    Uses per-format logistic regression models from Part 2 to
    predict execution success probability for each format,
    then routes to the format with highest probability.

    Args:
        affinity_models: Dict mapping format -> fitted AffinityModel.
        formats: List of candidate formats.
    """

    def __init__(
        self,
        affinity_models: dict[str, Any],  # format -> fitted sklearn model
        formats: list[str] | None = None,
    ):
        self.affinity_models = affinity_models
        self.formats = formats or list(affinity_models.keys())

    def route(self, features: TaskFeatures) -> str:
        """Route a single task to the best format.

        Args:
            features: Extracted structural features.

        Returns:
            Best format name.
        """
        raise NotImplementedError

    def route_with_scores(
        self, features: TaskFeatures
    ) -> dict[str, float]:
        """Get predicted success probability for each format.

        Args:
            features: Extracted structural features.

        Returns:
            Dict mapping format to predicted probability.
        """
        raise NotImplementedError

    def route_batch(
        self, features_list: list[TaskFeatures]
    ) -> list[str]:
        """Route a batch of tasks.

        Args:
            features_list: List of feature sets.

        Returns:
            List of format choices.
        """
        raise NotImplementedError

    def evaluate(
        self,
        features_list: list[TaskFeatures],
        ground_truth_best: list[str],
    ) -> dict[str, float]:
        """Evaluate routing accuracy and execution pass rate improvement.

        Args:
            features_list: Task features.
            ground_truth_best: Oracle best format per task.

        Returns:
            Dict with 'accuracy', 'exec_pass_rate', 'improvement_over_best_fixed'.
        """
        raise NotImplementedError
