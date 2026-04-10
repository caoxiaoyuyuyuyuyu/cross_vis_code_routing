"""Phase 2: Generation + rendering.

Steps:
1. Load dataset from Phase 1
2. Generate visual code with Qwen2.5-Coder-7B and 14B
3. Render all outputs (SVG/TikZ/Graphviz → PNG)
4. Record execution pass/fail for each generation
"""

import argparse
import logging

from src.data.dataset import CrossFormatDataset
from src.generation.generator import VisualCodeGenerator
from src.generation.renderer import Renderer
from src.utils.config import Config
from src.utils.io import save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = Config.from_yaml(config_path)

    # Load dataset
    dataset = CrossFormatDataset.load(f"{cfg.data.output_dir}/dataset.jsonl")
    logger.info(f"Loaded {len(dataset)} tasks")

    renderer = Renderer(
        output_dir=f"{cfg.output_dir}/rendered",
        dpi=cfg.render.output_dpi,
        svg_timeout=cfg.render.svg_timeout,
        tikz_timeout=cfg.render.tikz_timeout,
        graphviz_timeout=cfg.render.graphviz_timeout,
        tikz_max_workers=cfg.render.tikz_max_workers,
    )

    for model_name in [cfg.models.gen_model_7b, cfg.models.gen_model_14b]:
        model_short = model_name.split("/")[-1]
        logger.info(f"Generating with {model_short}")

        generator = VisualCodeGenerator(
            model_name=model_name,
            tensor_parallel=cfg.generation.vllm_tensor_parallel,
            gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
        )

        # Generate
        gen_results = generator.generate_batch(list(dataset))
        logger.info(f"Generated {len(gen_results)} outputs")

        # Render
        render_items = [
            (r.generated_code, r.target_format, r.task_id)
            for r in gen_results
        ]
        render_results = renderer.render_batch(render_items)

        pass_rate = sum(1 for r in render_results if r.success) / len(render_results)
        logger.info(f"{model_short} execution pass rate: {pass_rate:.3f}")

        # Save results
        save_jsonl(
            [{"task_id": r.task_id, "format": r.target_format, "model": model_short,
              "code": r.generated_code} for r in gen_results],
            f"{cfg.output_dir}/generations/{model_short}.jsonl",
        )
        save_jsonl(
            [{"task_id": r.task_id, "format": r.format, "success": r.success,
              "error": r.error_message, "image_path": r.image_path}
             for r in render_results],
            f"{cfg.output_dir}/render_results/{model_short}.jsonl",
        )

        generator.cleanup()

    logger.info("Phase 2 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Generation + rendering")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
