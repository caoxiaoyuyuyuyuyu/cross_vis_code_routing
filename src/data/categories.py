"""D003: Five mega-category system for cross-format dataset.

Merges fine-grained categories into 5 mega-categories based on
visual structure similarity. Every (category × format) cell targets ≥ 50.

Mega-categories and their visual characteristics:
- comparative: Discrete elements compared along a shared axis (bar/box/pie)
- relational: Continuous data mappings showing trends/correlations (line/scatter/heatmap)
- mathematical: Equation-driven rendering with specialized coordinate systems (3d/polar/function)
- structural: Nodes connected by edges, layout encodes relationships (flowchart/graph/network)
- compositional: Spatial arrangement of geometric primitives (shapes/icons/tables)
"""

# Fine-grained → mega-category mapping
MEGA_CATEGORY_MAP = {
    # ── comparative: discrete comparison / part-to-whole ──
    "bar": "comparative",
    "histogram": "comparative",
    "box": "comparative",
    "pie": "comparative",
    "waterfall": "comparative",
    "funnel": "comparative",
    "chart": "comparative",  # generic "chart" defaults to comparative

    # ── relational: continuous variable relationships ──
    "line": "relational",
    "area": "relational",
    "scatter": "relational",
    "heatmap": "relational",
    "map": "relational",       # spatial data mapping (D003 correction)
    "radar": "relational",     # multi-dimensional data viz (D003 correction)
    "sankey": "relational",    # flow quantities between categories

    # ── mathematical: equation/function-driven plots ──
    "3d": "mathematical",
    "math_function": "mathematical",
    "signal": "mathematical",
    "polar": "mathematical",   # polar-coordinate function plots (not radar)
    "ternary": "mathematical",
    "contour": "mathematical",

    # ── structural: node-edge / hierarchical / flow topology ──
    "flowchart": "structural",
    "graph": "structural",
    "sequence": "structural",
    "diagram": "structural",
    "gantt": "structural",
    "treemap": "structural",
    "network": "structural",

    # ── compositional: geometric primitives, non-data-driven ──
    "geometric": "compositional",
    "icon": "compositional",
    "illustration": "compositional",
    "table": "compositional",
    "gauge": "compositional",

    # ── catch-alls: reclassified by description keywords ──
    # These should ideally not appear after classifier improvements,
    # but if they do, route to compositional (smallest category).
    "other": "compositional",
    "other_chart": "compositional",
}

VALID_MEGA_CATEGORIES = [
    "comparative", "relational", "mathematical",
    "structural", "compositional",
]


def to_mega_category(fine_category: str) -> str:
    """Map a fine-grained category to its mega-category.

    Args:
        fine_category: Fine-grained category string.

    Returns:
        Mega-category string. Defaults to 'compositional' for unknown categories.
    """
    return MEGA_CATEGORY_MAP.get(fine_category, "compositional")
