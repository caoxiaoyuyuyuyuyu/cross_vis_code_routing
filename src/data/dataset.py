"""Unified dataset class for cross-format visual code generation tasks."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.data.vgbench_matcher import MatchedTask
from src.data.visplotbench_adapter import AdaptedTask

logger = logging.getLogger(__name__)


@dataclass
class TaskSample:
    """A single generation task in a specific format."""
    task_id: str
    description: str  # format-neutral description
    category: str  # e.g., "flowchart", "bar", "geometric"
    target_format: str  # "svg", "tikz", or "graphviz"
    prompt: str  # format-specific prompt for the LLM
    source: str  # "vgbench" or "visplotbench"
    eligible_formats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# Generation prompt templates for VGBench-matched tasks
FORMAT_PROMPT_TEMPLATES = {
    "svg": """Generate an SVG image based on the following description.

Description: {description}

Requirements:
- Output valid SVG code that renders correctly in a web browser
- Include proper viewBox, width, and height attributes
- Use appropriate SVG elements to faithfully represent the description

Output ONLY the SVG code, starting with <svg and ending with </svg>.""",

    "tikz": """Generate TikZ code based on the following description.

Description: {description}

Requirements:
- Output valid TikZ code that compiles with pdflatex
- Include \\begin{{tikzpicture}} and \\end{{tikzpicture}}
- Faithfully represent the description using appropriate TikZ commands

Output ONLY the TikZ code.""",

}


class CrossFormatDataset:
    """Unified dataset combining VGBench-matched and VisPlotBench-adapted tasks.

    Provides iteration over TaskSample instances, grouped by format or
    category as needed for generation, evaluation, and held-out splits.

    Args:
        tasks: List of TaskSample instances.
    """

    def __init__(self, tasks: list[TaskSample] | None = None):
        self.tasks = tasks or []

    @classmethod
    def from_sources(
        cls,
        matched_tasks: list[MatchedTask],
        adapted_tasks: list[AdaptedTask],
    ) -> "CrossFormatDataset":
        """Build dataset from VGBench matches and VisPlotBench adaptations.

        For VGBench: each matched task generates one TaskSample per
        eligible format. For VisPlotBench: each adapted task is already
        format-specific.

        Args:
            matched_tasks: Output of VGBenchMatcher.
            adapted_tasks: Output of VisPlotBenchAdapter.

        Returns:
            Unified CrossFormatDataset.
        """
        samples: list[TaskSample] = []

        # VGBench matched tasks -> one sample per (task, format) pair
        for mt in matched_tasks:
            for fmt in mt.eligible_formats:
                prompt = FORMAT_PROMPT_TEMPLATES.get(fmt, "").format(
                    description=mt.neutral_description
                )
                samples.append(TaskSample(
                    task_id=f"{mt.task_id}_{fmt}",
                    description=mt.neutral_description,
                    category=mt.category,
                    target_format=fmt,
                    prompt=prompt,
                    source="vgbench",
                    eligible_formats=mt.eligible_formats,
                    metadata={
                        "source_ids": mt.source_ids,
                        "similarity_scores": mt.similarity_scores,
                    },
                ))

        # VisPlotBench adapted tasks -> already format-specific
        for at in adapted_tasks:
            samples.append(TaskSample(
                task_id=at.task_id,
                description=at.description,
                category=at.category,
                target_format=at.target_format,
                prompt=at.prompt,
                source="visplotbench",
                eligible_formats=["svg", "tikz"],  # by design
                metadata={
                    "original_id": at.original_id,
                    "data_spec": at.data_spec,
                    "style_spec": at.style_spec,
                },
            ))

        logger.info(
            f"Built dataset: {len(samples)} samples "
            f"(VGBench: {sum(1 for s in samples if s.source == 'vgbench')}, "
            f"VisPlotBench: {sum(1 for s in samples if s.source == 'visplotbench')})"
        )
        return cls(samples)

    @classmethod
    def load(cls, path: str | Path) -> "CrossFormatDataset":
        """Load dataset from JSONL file."""
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tasks.append(TaskSample(**d))
        logger.info(f"Loaded {len(tasks)} samples from {path}")
        return cls(tasks)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for task in self.tasks:
                f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.tasks)} samples to {path}")

    def filter_by_format(self, fmt: str) -> "CrossFormatDataset":
        """Return subset containing only tasks for a given format."""
        return CrossFormatDataset([t for t in self.tasks if t.target_format == fmt])

    def filter_by_category(self, category: str) -> "CrossFormatDataset":
        """Return subset containing only tasks in a given category."""
        return CrossFormatDataset([t for t in self.tasks if t.category == category])

    def filter_by_source(self, source: str) -> "CrossFormatDataset":
        """Return subset from a specific data source."""
        return CrossFormatDataset([t for t in self.tasks if t.source == source])

    def get_categories(self) -> list[str]:
        """Return sorted list of unique categories."""
        return sorted(set(t.category for t in self.tasks))

    def get_formats(self) -> list[str]:
        """Return sorted list of unique formats."""
        return sorted(set(t.target_format for t in self.tasks))

    def cell_counts(self) -> dict[tuple[str, str], int]:
        """Count tasks per (category, format) cell."""
        counts: dict[tuple[str, str], int] = {}
        for t in self.tasks:
            key = (t.category, t.target_format)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def cell_count_report(self, min_count: int = 50) -> str:
        """Generate a formatted cell count report.

        Args:
            min_count: Threshold below which cells are flagged.

        Returns:
            Markdown-formatted report string.
        """
        counts = self.cell_counts()
        categories = self.get_categories()
        formats = self.get_formats()

        lines = ["## Dataset Cell Counts (Category × Format)", ""]

        # Header
        header = "| Category | " + " | ".join(formats) + " | Total |"
        sep = "|" + "|".join(["---"] * (len(formats) + 2)) + "|"
        lines.extend([header, sep])

        # Rows
        total_by_fmt: dict[str, int] = {f: 0 for f in formats}
        flagged_cells = []
        for cat in categories:
            row_total = 0
            cells = []
            for fmt in formats:
                c = counts.get((cat, fmt), 0)
                total_by_fmt[fmt] += c
                row_total += c
                flag = " ⚠" if c < min_count and c > 0 else ""
                if c < min_count and c > 0:
                    flagged_cells.append((cat, fmt, c))
                cells.append(f"{c}{flag}")
            lines.append(f"| {cat} | " + " | ".join(cells) + f" | {row_total} |")

        # Total row
        grand_total = sum(total_by_fmt.values())
        total_cells = [str(total_by_fmt[f]) for f in formats]
        lines.append(f"| **Total** | " + " | ".join(total_cells) + f" | **{grand_total}** |")

        # Flagged cells
        if flagged_cells:
            lines.extend(["", f"### ⚠ Cells below threshold ({min_count}):", ""])
            for cat, fmt, c in flagged_cells:
                lines.append(f"- {cat} × {fmt}: {c}")

        # Source breakdown
        vgb = sum(1 for t in self.tasks if t.source == "vgbench")
        vpb = sum(1 for t in self.tasks if t.source == "visplotbench")
        lines.extend([
            "", "### Source Breakdown", "",
            f"- VGBench matched: {vgb}",
            f"- VisPlotBench adapted: {vpb}",
            f"- Total: {grand_total}",
        ])

        return "\n".join(lines)

    def held_out_category_splits(
        self, n_folds: int | None = None
    ) -> Iterator[tuple["CrossFormatDataset", "CrossFormatDataset"]]:
        """Yield (train, test) splits where test is one held-out category.

        Args:
            n_folds: Number of folds. If None, equals number of categories.

        Yields:
            (train_dataset, test_dataset) tuples.
        """
        categories = self.get_categories()
        if n_folds is not None:
            categories = categories[:n_folds]

        for held_out_cat in categories:
            train = [t for t in self.tasks if t.category != held_out_cat]
            test = [t for t in self.tasks if t.category == held_out_cat]
            if test:
                yield CrossFormatDataset(train), CrossFormatDataset(test)

    def unique_tasks(self) -> int:
        """Count unique task descriptions (format-neutral)."""
        return len(set(t.description for t in self.tasks))

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_samples": len(self.tasks),
            "unique_tasks": self.unique_tasks(),
            "categories": self.get_categories(),
            "formats": self.get_formats(),
            "by_source": {
                "vgbench": sum(1 for t in self.tasks if t.source == "vgbench"),
                "visplotbench": sum(1 for t in self.tasks if t.source == "visplotbench"),
            },
            "cell_counts": {
                f"{cat}_{fmt}": count
                for (cat, fmt), count in sorted(self.cell_counts().items())
            },
        }

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[TaskSample]:
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> TaskSample:
        return self.tasks[idx]
