"""Phase 2: Generation + rendering (decoupled).

Step 1 (generate): Load dataset, generate code with Qwen2.5-Coder-7B via vLLM.
Step 2 (render): Load generated code, render SVG/TikZ to PNG.

Usage:
    # Full pipeline (generate + render)
    python scripts/run_phase2.py --config configs/default.yaml

    # Generate only (persist to JSON first)
    python scripts/run_phase2.py --config configs/default.yaml --step generate

    # Render only (load from saved generations)
    python scripts/run_phase2.py --config configs/default.yaml --step render
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from src.data.dataset import CrossFormatDataset
from src.utils.io import save_jsonl, load_jsonl, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def step_generate(cfg, model_name: str, dataset: CrossFormatDataset) -> Path:
    """Generate visual code for all tasks."""
    from src.generation.generator import VisualCodeGenerator

    model_short = model_name.split("/")[-1]
    out_dir = ensure_dir(f"{cfg.output_dir}/generations")
    out_path = out_dir / f"{model_short}.jsonl"

    logger.info(f"=" * 60)
    logger.info(f"Step 1: Generating with {model_short}")
    logger.info(f"  Tasks: {len(dataset)}")
    logger.info(f"  Max tokens: {cfg.generation.max_new_tokens}")
    logger.info(f"  Temperature: {cfg.generation.temperature}")
    logger.info(f"=" * 60)

    generator = VisualCodeGenerator(
        model_name=model_name,
        tensor_parallel=cfg.generation.vllm_tensor_parallel,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
    )

    gen_results = generator.generate_batch(list(dataset))
    logger.info(f"Generated {len(gen_results)} outputs")

    # Persist
    records = []
    for r in gen_results:
        records.append({
            "task_id": r.task_id,
            "target_format": r.target_format,
            "model": r.model_name,
            "generated_code": r.generated_code,
            "tokens_used": r.tokens_used,
            "metadata": r.metadata,
        })
    save_jsonl(records, out_path)
    logger.info(f"Saved generations to {out_path}")

    # Stats
    by_fmt = Counter(r.target_format for r in gen_results)
    avg_tokens = sum(r.tokens_used for r in gen_results) / len(gen_results)
    logger.info(f"  By format: {dict(by_fmt)}")
    logger.info(f"  Avg tokens: {avg_tokens:.0f}")

    generator.cleanup()
    return out_path


def step_render(cfg, model_short: str, dataset: CrossFormatDataset) -> Path:
    """Render generated code to PNG images."""
    from src.generation.renderer import Renderer

    gen_path = Path(f"{cfg.output_dir}/generations/{model_short}.jsonl")
    if not gen_path.exists():
        logger.error(f"Generations not found: {gen_path}")
        sys.exit(1)

    gen_records = load_jsonl(gen_path)
    logger.info(f"Loaded {len(gen_records)} generations from {gen_path}")

    out_dir = ensure_dir(f"{cfg.output_dir}/render_results")
    out_path = out_dir / f"{model_short}.jsonl"

    logger.info(f"=" * 60)
    logger.info(f"Step 2: Rendering {len(gen_records)} outputs")
    logger.info(f"  SVG timeout: {cfg.render.svg_timeout}s")
    logger.info(f"  TikZ timeout: {cfg.render.tikz_timeout}s")
    logger.info(f"  TikZ workers: {cfg.render.tikz_max_workers}")
    logger.info(f"=" * 60)

    renderer = Renderer(
        output_dir=f"{cfg.output_dir}/rendered/{model_short}",
        dpi=cfg.render.output_dpi,
        svg_timeout=cfg.render.svg_timeout,
        tikz_timeout=cfg.render.tikz_timeout,
        tikz_max_workers=cfg.render.tikz_max_workers,
    )

    items = [
        (r["generated_code"], r["target_format"], r["task_id"])
        for r in gen_records
    ]
    render_results = renderer.render_batch(items)

    # Persist
    records = []
    for r in render_results:
        records.append({
            "task_id": r.task_id,
            "format": r.format,
            "success": r.success,
            "error_message": r.error_message,
            "image_path": r.image_path,
            "render_time_seconds": r.render_time_seconds,
        })
    save_jsonl(records, out_path)
    logger.info(f"Saved render results to {out_path}")

    # Build category map from dataset
    cat_map = {t.task_id: t.category for t in dataset}

    # Success rate table: category × format
    _print_success_table(render_results, cat_map)

    return out_path


def _print_success_table(
    results: list, cat_map: dict[str, str]
) -> None:
    """Print execution success rate by category × format."""
    cells: dict[tuple[str, str], list[bool]] = {}
    for r in results:
        cat = cat_map.get(r.task_id, "unknown")
        key = (cat, r.format)
        cells.setdefault(key, []).append(r.success)

    categories = sorted(set(k[0] for k in cells))
    formats = sorted(set(k[1] for k in cells))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Execution Success Rate (Category × Format)")
    logger.info("=" * 60)

    header = "| Category | " + " | ".join(formats) + " | Total |"
    sep = "|" + "|".join(["---"] * (len(formats) + 2)) + "|"
    logger.info(header)
    logger.info(sep)

    total_success = 0
    total_count = 0
    for cat in categories:
        parts = []
        cat_success = 0
        cat_count = 0
        for fmt in formats:
            vals = cells.get((cat, fmt), [])
            s = sum(vals)
            n = len(vals)
            cat_success += s
            cat_count += n
            rate = f"{s}/{n} ({100*s/n:.0f}%)" if n > 0 else "—"
            parts.append(rate)
        total_success += cat_success
        total_count += cat_count
        total_rate = f"{cat_success}/{cat_count} ({100*cat_success/cat_count:.0f}%)"
        logger.info(f"| {cat} | " + " | ".join(parts) + f" | {total_rate} |")

    overall = f"{total_success}/{total_count} ({100*total_success/total_count:.0f}%)"
    logger.info(f"| **Total** | | | {overall} |")
    logger.info("=" * 60)

    # Log failures summary
    failures = [r for r in results if not r.success]
    if failures:
        err_counts = Counter()
        for f in failures:
            # Truncate error to first 80 chars for grouping
            err_key = (f.error_message or "unknown")[:80]
            err_counts[err_key] += 1
        logger.info(f"\nTop failure reasons ({len(failures)} total):")
        for err, cnt in err_counts.most_common(10):
            logger.info(f"  [{cnt}] {err}")


def main(config_path: str, step: str = "all", model: str = "7b") -> None:
    from src.utils.config import Config
    cfg = Config.from_yaml(config_path)

    # Select model
    if model == "7b":
        model_name = cfg.models.gen_model_7b
    elif model == "14b":
        model_name = cfg.models.gen_model_14b
    else:
        model_name = model  # allow full model path
    model_short = model_name.split("/")[-1]

    # Load dataset
    dataset_path = f"{cfg.data.output_dir}/dataset.jsonl"
    dataset = CrossFormatDataset.load(dataset_path)
    logger.info(f"Loaded dataset: {len(dataset)} samples")

    if step in ("all", "generate"):
        step_generate(cfg, model_name, dataset)

    if step in ("all", "render"):
        step_render(cfg, model_short, dataset)

    logger.info("Phase 2 done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Generation + rendering")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--step", choices=["all", "generate", "render"], default="all",
        help="Run generation, rendering, or both",
    )
    parser.add_argument(
        "--model", default="7b", choices=["7b", "14b"],
        help="Which model to use",
    )
    args = parser.parse_args()
    main(args.config, args.step, args.model)
