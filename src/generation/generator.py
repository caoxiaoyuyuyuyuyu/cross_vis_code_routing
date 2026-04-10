"""Visual code generation using Qwen2.5-Coder models via vLLM."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.data.dataset import TaskSample

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a single generation."""
    task_id: str
    target_format: str
    model_name: str
    generated_code: str
    prompt: str
    tokens_used: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_code(text: str, target_format: str) -> str:
    """Extract code block from LLM output.

    Handles markdown fenced blocks and raw output.
    """
    # Try markdown fenced code block first
    patterns = [
        rf"```(?:{target_format}|xml|latex|tex)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()

    # For SVG, extract <svg>...</svg>
    if target_format == "svg":
        m = re.search(r"(<svg[\s\S]*?</svg>)", text, re.DOTALL)
        if m:
            return m.group(1).strip()

    # For TikZ, extract \begin{tikzpicture}...\end{tikzpicture}
    # or \begin{axis}...\end{axis}
    if target_format == "tikz":
        for env in ["tikzpicture", "axis", "document"]:
            pat = rf"(\\begin\{{{env}\}}[\s\S]*?\\end\{{{env}\}})"
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()

    # Fallback: return full text stripped
    return text.strip()


class VisualCodeGenerator:
    """Generates visual code (SVG/TikZ) using vLLM.

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
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._llm = None
        self._tokenizer = None

    def _init_engine(self) -> None:
        """Initialize vLLM engine lazily."""
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # noqa: F401

        logger.info(
            f"Initializing vLLM: model={self.model_name}, "
            f"tp={self.tensor_parallel}, gpu_mem={self.gpu_memory_utilization}"
        )
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=8192,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def _build_prompt(self, task: TaskSample) -> str:
        """Build chat-formatted prompt for the model.

        Uses the task's pre-built format-specific prompt and wraps it
        in the chat template expected by Qwen2.5-Coder-Instruct.
        """
        messages = [{"role": "user", "content": task.prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(
        self, tasks: list[TaskSample], batch_size: int = 0
    ) -> list[GenerationResult]:
        """Generate visual code for a batch of tasks using vLLM.

        vLLM handles batching internally via continuous batching,
        so we submit all prompts at once.

        Args:
            tasks: List of task samples.
            batch_size: Ignored (vLLM handles batching internally).

        Returns:
            List of GenerationResult instances.
        """
        self._init_engine()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
        )

        # Build all prompts
        prompts = []
        for task in tasks:
            prompts.append(self._build_prompt(task))

        logger.info(f"Generating {len(prompts)} outputs with {self.model_name}")
        outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for task, output in zip(tasks, outputs):
            raw_text = output.outputs[0].text
            code = _extract_code(raw_text, task.target_format)
            tokens = len(output.outputs[0].token_ids)

            results.append(GenerationResult(
                task_id=task.task_id,
                target_format=task.target_format,
                model_name=self.model_name.split("/")[-1],
                generated_code=code,
                prompt=task.prompt,
                tokens_used=tokens,
                metadata={"raw_output_length": len(raw_text)},
            ))

        logger.info(f"Generation complete: {len(results)} results")
        return results

    def generate(self, task: TaskSample) -> GenerationResult:
        """Generate visual code for a single task."""
        return self.generate_batch([task])[0]

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
