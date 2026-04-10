"""SVG/TikZ/Graphviz rendering and execution checking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RenderResult:
    """Result of rendering a visual code snippet."""
    task_id: str
    format: str
    success: bool
    image_path: str | None  # path to rendered PNG
    error_message: str | None
    render_time_seconds: float
    metadata: dict[str, Any]


class Renderer:
    """Renders visual code to images and checks execution success.

    Supports SVG (via cairosvg/librsvg), TikZ (via pdflatex+pdf2image),
    and Graphviz (via graphviz python bindings).

    Args:
        output_dir: Directory to store rendered images.
        dpi: Resolution for rasterization.
        svg_timeout: Timeout for SVG rendering in seconds.
        tikz_timeout: Timeout for TikZ compilation in seconds.
        graphviz_timeout: Timeout for Graphviz rendering in seconds.
        tikz_max_workers: Number of parallel pdflatex workers.
    """

    def __init__(
        self,
        output_dir: str | Path,
        dpi: int = 300,
        svg_timeout: int = 10,
        tikz_timeout: int = 30,
        graphviz_timeout: int = 10,
        tikz_max_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.svg_timeout = svg_timeout
        self.tikz_timeout = tikz_timeout
        self.graphviz_timeout = graphviz_timeout
        self.tikz_max_workers = tikz_max_workers

    def render_svg(self, code: str, task_id: str) -> RenderResult:
        """Render SVG code to PNG.

        Args:
            code: SVG source code.
            task_id: Unique task identifier.

        Returns:
            RenderResult with success status and image path.
        """
        raise NotImplementedError

    def render_tikz(self, code: str, task_id: str) -> RenderResult:
        """Compile TikZ code via pdflatex and convert to PNG.

        Args:
            code: TikZ source code (with or without document preamble).
            task_id: Unique task identifier.

        Returns:
            RenderResult with success status and image path.
        """
        raise NotImplementedError

    def render_graphviz(self, code: str, task_id: str) -> RenderResult:
        """Render Graphviz DOT code to PNG.

        Args:
            code: Graphviz DOT source code.
            task_id: Unique task identifier.

        Returns:
            RenderResult with success status and image path.
        """
        raise NotImplementedError

    def render(self, code: str, fmt: str, task_id: str) -> RenderResult:
        """Render code in the specified format.

        Args:
            code: Source code.
            fmt: Format name ("svg", "tikz", or "graphviz").
            task_id: Unique task identifier.

        Returns:
            RenderResult.
        """
        dispatch = {
            "svg": self.render_svg,
            "tikz": self.render_tikz,
            "graphviz": self.render_graphviz,
        }
        if fmt not in dispatch:
            raise ValueError(f"Unknown format: {fmt}")
        return dispatch[fmt](code, task_id)

    def render_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[RenderResult]:
        """Render a batch of (code, format, task_id) tuples.

        Uses parallel workers for TikZ compilation.

        Args:
            items: List of (code, format, task_id) tuples.

        Returns:
            List of RenderResult instances.
        """
        raise NotImplementedError
