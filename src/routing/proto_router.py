"""VisRoute-Proto: prototype-based format routing.

Routes tasks to the format whose successful task prototypes are
nearest in embedding space. Supervision-light approach.
"""

from pathlib import Path

import numpy as np


class ProtoRouter:
    """Prototype routing via mean embeddings of successful task-format pairs.

    Computes per-format prototype embeddings from successfully executed
    tasks, then routes new tasks to the format whose prototype is
    closest in embedding space.

    Args:
        embed_model: Model for computing task embeddings.
        formats: List of candidate formats.
        k: Number of nearest neighbors for k-NN variant.
    """

    def __init__(
        self,
        embed_model: str = "Qwen/Qwen2.5-7B-Instruct",
        formats: list[str] | None = None,
        k: int = 5,
    ):
        self.embed_model = embed_model
        self.formats = formats or ["svg", "tikz", "graphviz"]
        self.k = k
        self._prototypes: dict[str, np.ndarray] = {}
        self._success_embeddings: dict[str, np.ndarray] = {}

    def build_prototypes(
        self,
        descriptions: list[str],
        formats: list[str],
        success: list[bool],
    ) -> None:
        """Build per-format prototypes from successful tasks.

        Args:
            descriptions: Task descriptions.
            formats: Format used for each task.
            success: Whether each task executed successfully.
        """
        raise NotImplementedError

    def route(self, description: str) -> str:
        """Route a single task to best format by prototype distance.

        Args:
            description: Task description.

        Returns:
            Best format name.
        """
        raise NotImplementedError

    def route_with_scores(self, description: str) -> dict[str, float]:
        """Get similarity scores to each format prototype.

        Args:
            description: Task description.

        Returns:
            Dict mapping format to similarity score.
        """
        raise NotImplementedError

    def route_batch(self, descriptions: list[str]) -> list[str]:
        """Route a batch of descriptions.

        Args:
            descriptions: List of task descriptions.

        Returns:
            List of format choices.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save prototypes to disk."""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load prototypes from disk."""
        raise NotImplementedError

    def evaluate(
        self,
        descriptions: list[str],
        ground_truth_best: list[str],
    ) -> dict[str, float]:
        """Evaluate routing accuracy.

        Args:
            descriptions: Task descriptions.
            ground_truth_best: Oracle best format per task.

        Returns:
            Evaluation metrics dict.
        """
        raise NotImplementedError
