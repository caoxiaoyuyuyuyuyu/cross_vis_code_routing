"""Unified mega-category system for cross-format dataset.

Merges fine-grained categories into 5-8 mega-categories to ensure
each (category × format) cell has ≥ 50 tasks.

Mega-category mapping:
- bar + histogram → bar_chart
- line + area → line_chart
- scatter + bubble → scatter_plot
- pie + donut → pie_chart
- heatmap + matrix → heatmap
- diagram + flowchart + graph + sequence + map + table → diagram
- geometric + icon + illustration → diagram (VGBench-sourced)
- remaining → other (target: ≤ 15% of total)
"""

# Fine-grained → mega-category mapping
MEGA_CATEGORY_MAP = {
    # Data visualization categories (primarily VisPlotBench)
    "bar": "bar_chart",
    "histogram": "bar_chart",
    "line": "line_chart",
    "area": "line_chart",
    "scatter": "scatter_plot",
    "pie": "pie_chart",
    "heatmap": "heatmap",
    "box": "other",
    "radar": "other",
    "treemap": "other",
    "sankey": "other",
    "gantt": "other",
    "waterfall": "other",
    "funnel": "other",
    "gauge": "other",
    "3d": "other",
    # Diagram categories (primarily VGBench)
    "flowchart": "diagram",
    "graph": "diagram",
    "sequence": "diagram",
    "diagram": "diagram",
    "geometric": "diagram",
    "icon": "diagram",
    "illustration": "diagram",
    "table": "other",
    "map": "other",
    "chart": "bar_chart",  # generic "chart" → bar_chart as most common
    # Catch-alls
    "other": "other",
    "other_chart": "other",
}

VALID_MEGA_CATEGORIES = [
    "bar_chart", "line_chart", "scatter_plot", "pie_chart",
    "heatmap", "diagram", "other",
]


def to_mega_category(fine_category: str) -> str:
    """Map a fine-grained category to its mega-category.

    Args:
        fine_category: Fine-grained category string.

    Returns:
        Mega-category string.
    """
    return MEGA_CATEGORY_MAP.get(fine_category, "other")
