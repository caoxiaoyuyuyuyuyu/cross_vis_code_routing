"""Format decoupling analysis: two-pass vs single-pass generation.

Tests whether decoupling format from task reasoning (à la Deco-G/Format Tax)
helps in visual code generation where format IS the representation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DecouplingResult:
    """Result of decoupling experiment for a single task."""
    task_id: str
    target_format: str
    single_pass_success: bool
    two_pass_success: bool
    pseudocode: str  # intermediate format-free pseudocode
    converted_code: str  # final format-specific code
    metadata: dict[str, Any]


class TwoPassGenerator:
    """Two-pass format decoupling generator.

    Pass 1: Generate format-free visual pseudocode.
    Pass 2: Convert pseudocode to target format.

    Compares against single-pass direct generation.

    Args:
        model_name: LLM model for both passes.
        pseudocode_max_tokens: Max tokens for pass 1.
        conversion_max_tokens: Max tokens for pass 2.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        pseudocode_max_tokens: int = 1024,
        conversion_max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.pseudocode_max_tokens = pseudocode_max_tokens
        self.conversion_max_tokens = conversion_max_tokens

    def _build_pseudocode_prompt(self, description: str) -> str:
        """Build prompt for format-free pseudocode generation.

        Args:
            description: Format-neutral task description.

        Returns:
            Prompt string.
        """
        raise NotImplementedError

    def _build_conversion_prompt(
        self, pseudocode: str, target_format: str
    ) -> str:
        """Build prompt for converting pseudocode to target format.

        Args:
            pseudocode: Format-free visual pseudocode.
            target_format: Target format (svg/tikz/graphviz).

        Returns:
            Prompt string.
        """
        raise NotImplementedError

    def generate_two_pass(
        self, description: str, target_format: str, task_id: str
    ) -> DecouplingResult:
        """Run two-pass generation pipeline.

        Args:
            description: Format-neutral task description.
            target_format: Target format.
            task_id: Unique task identifier.

        Returns:
            DecouplingResult with both passes' outputs.
        """
        raise NotImplementedError

    def generate_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[DecouplingResult]:
        """Batch two-pass generation.

        Args:
            items: List of (description, target_format, task_id).

        Returns:
            List of DecouplingResult instances.
        """
        raise NotImplementedError

    def compute_decoupling_effect(
        self,
        results: list[DecouplingResult],
    ) -> dict[str, Any]:
        """Compute decoupling effect size and test pre-registered hypothesis.

        Pre-registered: decoupling improves Execution Pass Rate by
        ≤ 50% of Format Tax's relative gain for general reasoning.

        Args:
            results: List of decoupling results.

        Returns:
            Dict with 'single_pass_rate', 'two_pass_rate', 'relative_gain',
            'threshold_met', 'conclusion'.
        """
        raise NotImplementedError
