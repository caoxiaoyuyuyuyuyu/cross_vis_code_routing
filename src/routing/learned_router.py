"""VisRoute-Learned: fine-tuned BERT format router.

Classifies task descriptions directly into the best output format
using a fine-tuned BERT model (~110M parameters).
"""

from pathlib import Path
from typing import Any


class LearnedRouter:
    """BERT-based format router trained on raw task descriptions.

    Fine-tunes BERT to classify tasks into the format that achieves
    highest execution pass rate.

    Args:
        model_name: Pre-trained BERT model identifier.
        formats: List of target format classes.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        formats: list[str] | None = None,
        lr: float = 2e-5,
        epochs: int = 5,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.formats = formats or ["svg", "tikz", "graphviz"]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def train(
        self,
        descriptions: list[str],
        labels: list[str],
        val_descriptions: list[str] | None = None,
        val_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train the BERT router.

        Args:
            descriptions: Task description texts.
            labels: Best format labels.
            val_descriptions: Optional validation descriptions.
            val_labels: Optional validation labels.

        Returns:
            Training metrics dict.
        """
        raise NotImplementedError

    def route(self, description: str) -> str:
        """Route a single task description to best format.

        Args:
            description: Task description text.

        Returns:
            Predicted best format.
        """
        raise NotImplementedError

    def route_with_scores(self, description: str) -> dict[str, float]:
        """Get predicted probability for each format.

        Args:
            description: Task description text.

        Returns:
            Dict mapping format to probability.
        """
        raise NotImplementedError

    def route_batch(self, descriptions: list[str]) -> list[str]:
        """Route a batch of descriptions.

        Args:
            descriptions: List of task descriptions.

        Returns:
            List of predicted formats.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load trained model from disk."""
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
            Dict with 'accuracy', 'exec_pass_rate', etc.
        """
        raise NotImplementedError
