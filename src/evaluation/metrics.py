"""Execution Pass Rate and metric aggregation."""

from dataclasses import dataclass
from typing import Any

from src.evaluation.vl_judge import JudgeResult
from src.generation.renderer import RenderResult


@dataclass
class AggregatedMetrics:
    """Aggregated evaluation metrics for a set of tasks."""
    execution_pass_rate: float
    visual_faithfulness_rate: float
    fid: float | None
    n_tasks: int
    per_category: dict[str, dict[str, float]]
    per_format: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def execution_pass_rate(results: list[RenderResult]) -> float:
    """Compute execution pass rate from render results.

    Args:
        results: List of RenderResult instances.

    Returns:
        Fraction of successfully rendered tasks.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def visual_faithfulness_rate(results: list[JudgeResult]) -> float:
    """Compute visual faithfulness pass rate.

    Args:
        results: List of JudgeResult instances.

    Returns:
        Fraction of tasks that pass visual faithfulness check.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.pass_fail) / len(results)


def aggregate_metrics(
    render_results: list[RenderResult],
    judge_results: list[JudgeResult],
    fid_scores: dict[str, float] | None = None,
    category_map: dict[str, str] | None = None,
) -> AggregatedMetrics:
    """Aggregate all metrics with per-category and per-format breakdowns.

    Args:
        render_results: Render results for execution pass rate.
        judge_results: Judge results for visual faithfulness.
        fid_scores: Optional per-format FID scores.
        category_map: Optional task_id -> category mapping.

    Returns:
        AggregatedMetrics with breakdowns.
    """
    raise NotImplementedError


def format_results_table(metrics: AggregatedMetrics) -> str:
    """Format metrics as a markdown table for reporting.

    Args:
        metrics: Aggregated metrics.

    Returns:
        Markdown-formatted results table.
    """
    raise NotImplementedError
