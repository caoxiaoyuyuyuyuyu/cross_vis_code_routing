"""SVG/TikZ rendering and execution checking."""

import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """Result of rendering a visual code snippet."""
    task_id: str
    format: str
    success: bool
    image_path: str | None = None
    error_message: str | None = None
    render_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Renderer:
    """Renders visual code to PNG images.

    Supports SVG (via cairosvg) and TikZ (via pdflatex + pdf2image).

    Args:
        output_dir: Directory to store rendered images.
        dpi: Resolution for rasterization.
        svg_timeout: Timeout for SVG rendering in seconds.
        tikz_timeout: Timeout for TikZ compilation in seconds.
        tikz_max_workers: Number of parallel pdflatex workers.
    """

    def __init__(
        self,
        output_dir: str | Path,
        dpi: int = 300,
        svg_timeout: int = 10,
        tikz_timeout: int = 30,
        tikz_max_workers: int = 4,
        **kwargs,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.svg_timeout = svg_timeout
        self.tikz_timeout = tikz_timeout
        self.tikz_max_workers = tikz_max_workers

    def render_svg(self, code: str, task_id: str) -> RenderResult:
        """Render SVG code to PNG via cairosvg."""
        t0 = time.time()
        out_path = self.output_dir / "svg" / f"{task_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import cairosvg
            cairosvg.svg2png(
                bytestring=code.encode("utf-8"),
                write_to=str(out_path),
                output_width=800,
                dpi=self.dpi,
            )
            return RenderResult(
                task_id=task_id, format="svg", success=True,
                image_path=str(out_path),
                render_time_seconds=time.time() - t0,
            )
        except Exception as e:
            return RenderResult(
                task_id=task_id, format="svg", success=False,
                error_message=str(e)[:500],
                render_time_seconds=time.time() - t0,
            )

    def render_tikz(self, code: str, task_id: str) -> RenderResult:
        """Compile TikZ code via pdflatex and convert to PNG."""
        t0 = time.time()
        out_path = self.output_dir / "tikz" / f"{task_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Wrap in document if needed
        if "\\documentclass" not in code:
            code = _wrap_tikz_document(code)

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "main.tex")
            pdf_path = os.path.join(tmpdir, "main.pdf")

            with open(tex_path, "w") as f:
                f.write(code)

            # Compile with pdflatex
            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-halt-on-error", "-output-directory", tmpdir, tex_path],
                    capture_output=True, text=True,
                    timeout=self.tikz_timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return RenderResult(
                    task_id=task_id, format="tikz", success=False,
                    error_message=f"pdflatex timeout ({self.tikz_timeout}s)",
                    render_time_seconds=time.time() - t0,
                )

            if not os.path.exists(pdf_path):
                err = result.stderr[-500:] if result.stderr else result.stdout[-500:]
                return RenderResult(
                    task_id=task_id, format="tikz", success=False,
                    error_message=f"pdflatex failed: {err}",
                    render_time_seconds=time.time() - t0,
                )

            # Convert PDF to PNG
            try:
                subprocess.run(
                    ["pdftoppm", "-png", "-r", str(self.dpi),
                     "-singlefile", pdf_path, str(out_path.with_suffix(""))],
                    capture_output=True, timeout=10,
                )
                if not out_path.exists():
                    return RenderResult(
                        task_id=task_id, format="tikz", success=False,
                        error_message="pdftoppm produced no output",
                        render_time_seconds=time.time() - t0,
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return RenderResult(
                    task_id=task_id, format="tikz", success=False,
                    error_message=f"pdf2png failed: {e}",
                    render_time_seconds=time.time() - t0,
                )

        return RenderResult(
            task_id=task_id, format="tikz", success=True,
            image_path=str(out_path),
            render_time_seconds=time.time() - t0,
        )

    def render(self, code: str, fmt: str, task_id: str) -> RenderResult:
        """Render code in the specified format."""
        dispatch = {
            "svg": self.render_svg,
            "tikz": self.render_tikz,
        }
        if fmt not in dispatch:
            return RenderResult(
                task_id=task_id, format=fmt, success=False,
                error_message=f"Unsupported format: {fmt}",
            )
        return dispatch[fmt](code, task_id)

    def render_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[RenderResult]:
        """Render a batch of (code, format, task_id) tuples.

        SVG rendering is fast (single-threaded sequential).
        TikZ uses parallel workers for pdflatex compilation.
        """
        svg_items = [(c, f, t) for c, f, t in items if f == "svg"]
        tikz_items = [(c, f, t) for c, f, t in items if f == "tikz"]

        results = {}

        # SVG: sequential (fast, I/O bound)
        for i, (code, fmt, task_id) in enumerate(svg_items):
            results[task_id] = self.render_svg(code, task_id)
            if (i + 1) % 100 == 0:
                logger.info(f"SVG rendered: {i+1}/{len(svg_items)}")
        if svg_items:
            logger.info(f"SVG rendering complete: {len(svg_items)} items")

        # TikZ: parallel (CPU-bound pdflatex)
        if tikz_items:
            logger.info(
                f"TikZ rendering: {len(tikz_items)} items, "
                f"{self.tikz_max_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=self.tikz_max_workers) as pool:
                futures = {
                    pool.submit(self.render_tikz, code, task_id): task_id
                    for code, _, task_id in tikz_items
                }
                done = 0
                for future in as_completed(futures):
                    task_id = futures[future]
                    results[task_id] = future.result()
                    done += 1
                    if done % 50 == 0:
                        logger.info(f"TikZ rendered: {done}/{len(tikz_items)}")
            logger.info(f"TikZ rendering complete: {len(tikz_items)} items")

        # Return in original order
        return [results[t] for _, _, t in items]


def _wrap_tikz_document(tikz_code: str) -> str:
    """Wrap TikZ code in a minimal LaTeX document."""
    return (
        "\\documentclass[border=5pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\begin{document}\n"
        f"{tikz_code}\n"
        "\\end{document}\n"
    )
