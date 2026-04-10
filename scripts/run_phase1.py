"""Phase 1: Dataset construction.

Steps:
1. VGBench cross-format semantic matching → 300-500 format-neutral tasks
2. VisPlotBench SVG+TikZ adaptation → 600-800 tasks
3. Combine and audit per-cell counts (category × format ≥ 50)
4. Save unified dataset + cell count report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.data.dataset import CrossFormatDataset
from src.data.visplotbench_adapter import VisPlotBenchAdapter
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = Config.from_yaml(config_path)
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────── Step 1: VisPlotBench adaptation ────────────────
    logger.info("=" * 60)
    logger.info("Step 1: VisPlotBench SVG+TikZ adaptation (D006: visplotbench-only)")
    logger.info("=" * 60)

    adapter = VisPlotBenchAdapter(visplotbench_dir=cfg.data.visplotbench_dir)
    adapted_tasks = adapter.run()
    logger.info(f"VisPlotBench adapted tasks: {len(adapted_tasks)}")

    # Save intermediate results
    adapted_path = output_dir / "visplotbench_adapted.json"
    with open(adapted_path, "w") as f:
        from dataclasses import asdict
        json.dump([asdict(a) for a in adapted_tasks], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved VisPlotBench adaptations to {adapted_path}")

    # ──────────────── Step 2: Build dataset + Audit ────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Building unified dataset + cell audit")
    logger.info("=" * 60)

    dataset = CrossFormatDataset.from_sources([], adapted_tasks)

    # Cell count report
    report = dataset.cell_count_report(min_count=cfg.data.min_cell_count)
    logger.info("\n" + report)

    # Save report
    report_path = output_dir / "cell_count_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Cell count report saved to {report_path}")

    # Summary
    summary = dataset.summary()
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ──────────────── Step 3: Save dataset ────────────────
    dataset_path = output_dir / "dataset.jsonl"
    dataset.save(dataset_path)
    logger.info(f"Dataset saved to {dataset_path}")

    # Final summary
    logger.info("=" * 60)
    logger.info("Phase 1 Complete")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Unique tasks: {dataset.unique_tasks()}")
    logger.info(f"  Categories: {dataset.get_categories()}")
    logger.info(f"  Formats: {dataset.get_formats()}")
    logger.info(f"  VisPlotBench: {summary['by_source']['visplotbench']}")

    # Check if we meet the minimum threshold
    cell_counts = dataset.cell_counts()
    below_threshold = [
        (cat, fmt, c)
        for (cat, fmt), c in cell_counts.items()
        if 0 < c < cfg.data.min_cell_count
    ]
    if below_threshold:
        logger.warning(
            f"⚠ {len(below_threshold)} cells below minimum count "
            f"({cfg.data.min_cell_count}). Consider adjusting threshold "
            f"or restricting affinity modeling."
        )

    total_unique = dataset.unique_tasks()
    if total_unique < 600:
        logger.warning(
            f"⚠ Total unique tasks ({total_unique}) < 600. "
            f"Hard fallback: consider dropping Graphviz for 2-format study."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Dataset construction")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
