"""VisPlotBench → SVG + TikZ adaptation.

Adapts VisPlotBench tasks (originally multi-language data visualization)
into SVG and TikZ prompts. Graphviz is excluded by design since
data-viz tasks (bar/line/scatter) are not naturally Graphviz-expressible.

Data source: VisCoder2 release (888 tasks across 8 visual languages).
HuggingFace: likely `VisCoder/VisPlotBench` or from VisCoder2 repo.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# VisPlotBench task categories — expanded keyword lists + metadata-aware
VISPLOT_CATEGORIES = {
    "bar": ["bar chart", "bar graph", "bar plot", "grouped bar", "stacked bar",
            "horizontal bar", "vertical bar", "column chart", "column graph",
            "barplot", "barchart", "barh"],
    "line": ["line chart", "line graph", "line plot", "time series", "trend line",
             "multi-line", "sparkline", "lineplot", "linechart", "curve chart",
             "step chart", "step plot"],
    "scatter": ["scatter plot", "scatter chart", "scatter graph", "bubble chart",
                "bubble plot", "scatterplot", "point plot", "dot plot",
                "regression plot", "lmplot"],
    "pie": ["pie chart", "pie graph", "donut chart", "donut graph",
            "ring chart", "piechart"],
    "heatmap": ["heatmap", "heat map", "matrix plot", "correlation matrix",
                "confusion matrix", "clustermap", "color map", "grid plot"],
    "area": ["area chart", "area graph", "stacked area", "filled area",
             "area plot", "stream graph", "streamgraph"],
    "histogram": ["histogram", "distribution plot", "frequency distribution",
                  "density plot", "kde", "distplot", "hist plot"],
    "box": ["box plot", "boxplot", "violin plot", "violinplot", "whisker",
            "box-and-whisker", "swarm plot", "strip plot"],
    "radar": ["radar chart", "spider chart", "polar chart", "star chart",
              "radial chart", "polar plot", "wind rose"],
    "table": ["table", "tabular", "grid", "spreadsheet", "data table"],
    "map": ["map", "choropleth", "geographic", "geospatial", "cartogram",
            "contour map", "topographic"],
    "treemap": ["treemap", "tree map", "sunburst", "icicle chart",
                "partition plot"],
    "sankey": ["sankey", "alluvial", "flow diagram", "parallel coordinates",
               "parallel categories"],
    "gantt": ["gantt", "timeline chart", "schedule chart", "project timeline"],
    "waterfall": ["waterfall chart", "waterfall plot", "bridge chart"],
    "funnel": ["funnel chart", "funnel plot", "pipeline chart"],
    "gauge": ["gauge chart", "gauge plot", "speedometer", "dial chart"],
    "3d": ["3d plot", "3d chart", "surface plot", "wireframe", "3d scatter",
           "3d bar", "3d surface", "contour plot", "contour chart"],
}

# Mapping from TIGER-Lab/VisPlotBench _config + task metadata to our categories
# The _config field tells us the original visualization language;
# task__plot_type (if present) gives the chart type directly.
_CONFIG_CATEGORY_HINTS = {
    "mermaid": "diagram",  # mermaid tasks are mostly diagrams/flowcharts
}


def _classify_visplot_category(task: dict[str, Any]) -> str:
    """Classify a VisPlotBench task into category.

    Strategy (ordered by reliability):
    1. Use task__plot_type metadata if present (most reliable)
    2. Use _config field hints for non-chart languages (mermaid → diagram)
    3. Keyword matching against description text
    """
    # Strategy 1: explicit plot_type metadata from VisPlotBench
    for meta_field in ["task__plot_type", "plot_type", "chart_type", "type"]:
        plot_type = task.get(meta_field, "")
        if isinstance(plot_type, str) and plot_type.strip():
            pt = plot_type.strip().lower()
            # Direct match against our categories
            for cat in VISPLOT_CATEGORIES:
                if cat in pt:
                    return cat
            # Common plot_type values → category mapping
            type_map = {
                "bar": "bar", "column": "bar", "grouped_bar": "bar",
                "stacked_bar": "bar", "horizontal_bar": "bar",
                "line": "line", "multi_line": "line", "step": "line",
                "scatter": "scatter", "bubble": "scatter", "point": "scatter",
                "pie": "pie", "donut": "pie", "ring": "pie",
                "heatmap": "heatmap", "matrix": "heatmap", "correlation": "heatmap",
                "area": "area", "stacked_area": "area", "stream": "area",
                "histogram": "histogram", "distribution": "histogram",
                "density": "histogram", "kde": "histogram",
                "box": "box", "violin": "box", "boxplot": "box",
                "radar": "radar", "spider": "radar", "polar": "radar",
                "table": "table",
                "map": "map", "choropleth": "map", "geo": "map",
                "treemap": "treemap", "sunburst": "treemap",
                "sankey": "sankey", "alluvial": "sankey",
                "gantt": "gantt", "timeline": "gantt",
                "waterfall": "waterfall",
                "funnel": "funnel",
                "contour": "3d", "surface": "3d", "3d": "3d",
            }
            for key, cat in type_map.items():
                if key in pt.replace(" ", "_"):
                    return cat

    # Strategy 2: _config field hints
    config = task.get("_config", "")
    if config in _CONFIG_CATEGORY_HINTS:
        return _CONFIG_CATEGORY_HINTS[config]

    # Strategy 3: keyword matching against description
    desc = ""
    for field_name in ["task__plot_description", "task", "Task",
                       "task_description", "description"]:
        val = task.get(field_name, "")
        if isinstance(val, str) and val.strip():
            desc = val.strip().lower()
            break

    if not desc:
        return "other_chart"

    scores: dict[str, int] = {}
    for cat, keywords in VISPLOT_CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in desc)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other_chart"

# Format-specific prompt templates
SVG_PROMPT_TEMPLATE = """Generate an SVG image for the following visualization task.

Task: {task_description}
Style: {style_description}

Requirements:
- Output valid SVG code that can be rendered in a web browser
- Include proper viewBox, width, and height attributes
- Use appropriate SVG elements (rect, circle, line, path, text, etc.)
- Add axis labels, title, and legend as specified
- Use the colors and styling described above

Output ONLY the SVG code, starting with <svg and ending with </svg>."""

TIKZ_PROMPT_TEMPLATE = """Generate TikZ code for the following visualization task.

Task: {task_description}
Style: {style_description}

Requirements:
- Output valid TikZ code that compiles with pdflatex
- Include \\begin{{tikzpicture}} and \\end{{tikzpicture}}
- Use pgfplots for data visualization if appropriate
- Add axis labels, title, and legend as specified
- Use the colors and styling described above

Output ONLY the TikZ code, starting with \\begin{{tikzpicture}} or \\begin{{axis}}."""


@dataclass
class AdaptedTask:
    """A VisPlotBench task adapted for SVG/TikZ generation."""
    task_id: str
    original_id: str
    description: str  # format-neutral description
    category: str  # bar, line, scatter, etc.
    target_format: str  # "svg" or "tikz"
    prompt: str  # format-specific prompt for the LLM
    data_spec: dict[str, Any] = field(default_factory=dict)
    style_spec: dict[str, Any] = field(default_factory=dict)


def _classify_visplot_category(text: str) -> str:
    """Classify a VisPlotBench task into category."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in VISPLOT_CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other_chart"


class VisPlotBenchAdapter:
    """Adapts VisPlotBench tasks to SVG and TikZ format prompts.

    VisPlotBench contains 888 tasks across 8 visual languages.
    We extract Task + Style Description fields and generate
    format-specific prompts for SVG and TikZ only.

    Args:
        visplotbench_dir: Path to VisPlotBench dataset directory.
    """

    def __init__(self, visplotbench_dir: str | Path):
        self.visplotbench_dir = Path(visplotbench_dir)

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load raw VisPlotBench tasks.

        Tries HuggingFace first, falls back to local files.

        Returns:
            List of task dicts from VisPlotBench.
        """
        tasks = []

        # Try HuggingFace
        # TIGER-Lab/VisPlotBench is split into per-language configs
        # (asymptote/html/latex/lilypond/mermaid/python/svg/vegalite);
        # each config has a single "test" split. We load all adaptable
        # configs and skip lilypond (audio/music only — not a plot).
        try:
            from datasets import load_dataset
            hf_path = "TIGER-Lab/VisPlotBench"
            configs = ["asymptote", "html", "latex", "mermaid",
                       "python", "svg", "vegalite"]
            for cfg in configs:
                try:
                    logger.info(f"Loading {hf_path} config={cfg}")
                    ds = load_dataset(hf_path, name=cfg, split="test")
                    for item in ds:
                        d = {k: v for k, v in item.items() if k != "image"}
                        d["_config"] = cfg
                        tasks.append(d)
                except Exception as e:
                    logger.warning(f"Skipping config {cfg}: {e}")
                    continue
            if tasks:
                logger.info(f"Loaded {len(tasks)} tasks from {hf_path}")
                return tasks
        except ImportError:
            pass

        # Fallback: local files
        logger.info(f"Loading VisPlotBench from local: {self.visplotbench_dir}")
        for ext in ["*.json", "*.jsonl"]:
            for fpath in sorted(self.visplotbench_dir.glob(ext)):
                if ext == "*.jsonl":
                    with open(fpath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                tasks.append(json.loads(line))
                else:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        tasks.extend(data)
                    elif isinstance(data, dict):
                        for key, item in data.items():
                            if isinstance(item, dict):
                                item.setdefault("id", key)
                                tasks.append(item)

        logger.info(f"Loaded {len(tasks)} tasks from local files")
        return tasks

    def _extract_task_description(self, task: dict[str, Any]) -> str:
        """Extract the format-neutral task description.

        Handles various field naming conventions.

        Args:
            task: Raw task dict.

        Returns:
            Task description string.
        """
        # Try common field names. `task__plot_description` is TIGER-Lab/VisPlotBench's name.
        for field_name in ["task__plot_description", "task", "Task",
                           "task_description", "description",
                           "instruction", "prompt", "query"]:
            val = task.get(field_name, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _extract_style_description(self, task: dict[str, Any]) -> str:
        """Extract style description from task.

        Args:
            task: Raw task dict.

        Returns:
            Style description string.
        """
        # `task__plot_style` is TIGER-Lab/VisPlotBench's field name.
        for field_name in ["task__plot_style", "style_description",
                           "Style Description", "style", "Style", "style_desc"]:
            val = task.get(field_name, "")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return "Use default styling with clear, readable fonts and appropriate colors."

    def _extract_data_spec(self, task: dict[str, Any]) -> dict[str, Any]:
        """Extract embedded data specification from task.

        Args:
            task: Raw task dict.

        Returns:
            Data specification dict.
        """
        data_spec = {}
        for field_name in ["data", "Data", "data_spec", "dataset", "values"]:
            val = task.get(field_name)
            if val is not None:
                data_spec["data"] = val
                break
        return data_spec

    def filter_adaptable(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter tasks that can be meaningfully adapted to SVG/TikZ.

        Removes tasks:
        - With empty descriptions
        - Too tightly coupled to interactive features (Vega-Lite interactions)
        - With only audio/music content (LilyPond-specific)

        Args:
            tasks: Raw task list.

        Returns:
            Filtered task list.
        """
        filtered = []
        skip_keywords = [
            "interactive", "hover", "tooltip", "click event",
            "animation", "transition", "lilypond", "music", "note",
            "midi", "audio", "sound",
        ]

        for task in tasks:
            desc = self._extract_task_description(task)
            if not desc:
                continue

            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in skip_keywords):
                continue

            filtered.append(task)

        logger.info(f"Filtered adaptable tasks: {len(tasks)} -> {len(filtered)}")
        return filtered

    def adapt_to_svg(self, task: dict[str, Any], task_idx: int) -> AdaptedTask:
        """Convert a VisPlotBench task to SVG generation prompt.

        Args:
            task: Raw VisPlotBench task dict.
            task_idx: Index for task ID generation.

        Returns:
            AdaptedTask with SVG-specific prompt.
        """
        desc = self._extract_task_description(task)
        style = self._extract_style_description(task)
        data_spec = self._extract_data_spec(task)

        prompt = SVG_PROMPT_TEMPLATE.format(
            task_description=desc,
            style_description=style,
        )

        original_id = str(task.get("id", task.get("idx", task_idx)))

        return AdaptedTask(
            task_id=f"vpb_svg_{task_idx:04d}",
            original_id=original_id,
            description=desc,
            category=_classify_visplot_category(task),
            target_format="svg",
            prompt=prompt,
            data_spec=data_spec,
            style_spec={"raw_style": style},
        )

    def adapt_to_tikz(self, task: dict[str, Any], task_idx: int) -> AdaptedTask:
        """Convert a VisPlotBench task to TikZ generation prompt.

        Args:
            task: Raw VisPlotBench task dict.
            task_idx: Index for task ID generation.

        Returns:
            AdaptedTask with TikZ-specific prompt.
        """
        desc = self._extract_task_description(task)
        style = self._extract_style_description(task)
        data_spec = self._extract_data_spec(task)

        prompt = TIKZ_PROMPT_TEMPLATE.format(
            task_description=desc,
            style_description=style,
        )

        original_id = str(task.get("id", task.get("idx", task_idx)))

        return AdaptedTask(
            task_id=f"vpb_tikz_{task_idx:04d}",
            original_id=original_id,
            description=desc,
            category=_classify_visplot_category(task),
            target_format="tikz",
            prompt=prompt,
            data_spec=data_spec,
            style_spec={"raw_style": style},
        )

    def run(self) -> list[AdaptedTask]:
        """Full pipeline: load → filter → adapt to SVG + TikZ.

        Returns:
            List of adapted tasks (2 per original: SVG + TikZ).
        """
        raw_tasks = self.load_tasks()
        adaptable = self.filter_adaptable(raw_tasks)

        adapted: list[AdaptedTask] = []
        for i, task in enumerate(adaptable):
            adapted.append(self.adapt_to_svg(task, i))
            adapted.append(self.adapt_to_tikz(task, i))

        logger.info(
            f"Adapted {len(adaptable)} tasks -> {len(adapted)} "
            f"(SVG: {len(adapted)//2}, TikZ: {len(adapted)//2})"
        )

        # Category distribution
        cat_counts: dict[str, int] = {}
        for a in adapted:
            cat_counts[a.category] = cat_counts.get(a.category, 0) + 1
        logger.info(f"Category distribution: {cat_counts}")

        return adapted
