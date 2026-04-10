"""Visual faithfulness judge using Qwen2.5-VL models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JudgeResult:
    """Result of visual faithfulness judgment."""
    task_id: str
    pass_fail: bool  # binary pass/fail
    score: float  # continuous score [0, 1]
    explanation: str
    model_name: str
    metadata: dict[str, Any]


class VLJudge:
    """LLM-as-Judge for visual faithfulness evaluation.

    Uses Qwen2.5-VL models to assess whether a rendered image
    faithfully represents the task description, following
    VGBench/VisCoder2 evaluation protocols.

    Args:
        model_name: VL model identifier (7B or 32B).
        device: Device for inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _init_model(self) -> None:
        """Load model and processor lazily."""
        raise NotImplementedError

    def _build_judge_prompt(
        self, description: str, image_path: str | Path
    ) -> Any:
        """Build multimodal prompt for faithfulness judgment.

        Args:
            description: Task description.
            image_path: Path to rendered image.

        Returns:
            Model input for generation.
        """
        raise NotImplementedError

    def judge(self, description: str, image_path: str | Path, task_id: str) -> JudgeResult:
        """Judge a single rendered image against its task description.

        Args:
            description: Format-neutral task description.
            image_path: Path to rendered PNG.
            task_id: Unique task identifier.

        Returns:
            JudgeResult with pass/fail and score.
        """
        raise NotImplementedError

    def judge_batch(
        self,
        items: list[tuple[str, str | Path, str]],
    ) -> list[JudgeResult]:
        """Judge a batch of (description, image_path, task_id) tuples.

        Args:
            items: List of (description, image_path, task_id).

        Returns:
            List of JudgeResult instances.
        """
        raise NotImplementedError

    def compute_inter_judge_agreement(
        self,
        judge_7b_results: list[JudgeResult],
        judge_32b_results: list[JudgeResult],
    ) -> dict[str, float]:
        """Compute Cohen's kappa between 7B and 32B judges.

        Args:
            judge_7b_results: Results from VL-7B judge.
            judge_32b_results: Results from VL-32B judge.

        Returns:
            Dict with 'kappa', 'agreement_rate', 'n_samples'.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release GPU resources."""
        raise NotImplementedError
