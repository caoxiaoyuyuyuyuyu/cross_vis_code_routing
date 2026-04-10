"""Visual code generation using Qwen2.5-Coder models via vLLM."""

from dataclasses import dataclass
from typing import Any

from src.data.dataset import TaskSample


@dataclass
class GenerationResult:
    """Result of a single generation."""
    task_id: str
    target_format: str
    model_name: str
    generated_code: str
    prompt: str
    tokens_used: int
    metadata: dict[str, Any]


class VisualCodeGenerator:
    """Generates visual code (SVG/TikZ/Graphviz) using vLLM.

    Supports Qwen2.5-Coder-7B-Instruct and 14B-Instruct.

    Args:
        model_name: HuggingFace model identifier.
        tensor_parallel: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._engine = None

    def _init_engine(self) -> None:
        """Initialize vLLM engine lazily."""
        raise NotImplementedError

    def _build_prompt(self, task: TaskSample) -> str:
        """Build format-specific generation prompt.

        Args:
            task: Task sample with description and target format.

        Returns:
            Formatted prompt string for the chat model.
        """
        raise NotImplementedError

    def generate(self, task: TaskSample) -> GenerationResult:
        """Generate visual code for a single task.

        Args:
            task: Task sample.

        Returns:
            GenerationResult with generated code.
        """
        raise NotImplementedError

    def generate_batch(self, tasks: list[TaskSample]) -> list[GenerationResult]:
        """Generate visual code for a batch of tasks.

        Args:
            tasks: List of task samples.

        Returns:
            List of GenerationResult instances.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release GPU resources."""
        raise NotImplementedError
