"""Validate MiniLM matching quality before lowering threshold.

Samples 20 matched pairs from VGBench cross-format matching,
prints source/target descriptions, and uses LLM to judge
semantic consistency (binary match/no-match).

Usage:
    python scripts/validate_matching.py --config configs/default.yaml
    python scripts/validate_matching.py --threshold 0.85  # override
"""

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml", threshold: float | None = None,
         n_samples: int = 20, seed: int = 42) -> None:
    from src.data.vgbench_matcher import VGBenchMatcher
    from src.utils.config import Config

    cfg = Config.from_yaml(config_path)

    # Override threshold if specified
    actual_threshold = threshold if threshold is not None else cfg.data.similarity_threshold
    logger.info(f"Running VGBench matching at threshold={actual_threshold}")

    matcher = VGBenchMatcher(
        vgbench_dir=cfg.data.vgbench_dir,
        embed_model=cfg.models.embed_model,
        similarity_threshold=actual_threshold,
    )

    # Load and match
    tasks_by_format = matcher.load_tasks()
    matched = matcher.match_across_formats(tasks_by_format)
    logger.info(f"Total matched pairs: {len(matched)}")

    if len(matched) == 0:
        logger.error("No matches found. Check data and threshold.")
        return

    # Sample pairs for validation
    random.seed(seed)
    sample_size = min(n_samples, len(matched))
    sampled = random.sample(matched, sample_size)

    # Collect source descriptions for each pair
    # For each matched task, get the caption from each format
    all_tasks_flat: dict[str, dict] = {}
    for fmt, tasks in tasks_by_format.items():
        for t in tasks:
            if t.get("caption", "").strip():
                all_tasks_flat[f"{fmt}_{t['idx']}"] = t

    print("\n" + "=" * 80)
    print(f"MiniLM MATCHING QUALITY VALIDATION")
    print(f"Threshold: {actual_threshold} | Total matches: {len(matched)} | Sampled: {sample_size}")
    print("=" * 80)

    pairs_for_llm = []
    for i, mt in enumerate(sampled):
        print(f"\n--- Pair {i+1}/{sample_size} (ID: {mt.task_id}) ---")
        print(f"Eligible formats: {mt.eligible_formats}")
        print(f"Similarity scores: {mt.similarity_scores}")
        print(f"Category: {mt.category}")

        # Print descriptions from each format
        for fmt, src_idx in mt.source_ids.items():
            key = f"{fmt}_{src_idx}"
            task_data = all_tasks_flat.get(key, {})
            caption = task_data.get("caption", "[NOT FOUND]")
            print(f"  [{fmt.upper()}] {caption[:200]}")

        # Collect for LLM validation
        descs = []
        for fmt in sorted(mt.source_ids.keys()):
            key = f"{fmt}_{mt.source_ids[fmt]}"
            task_data = all_tasks_flat.get(key, {})
            descs.append(task_data.get("caption", ""))
        pairs_for_llm.append({
            "task_id": mt.task_id,
            "descriptions": descs,
            "formats": sorted(mt.source_ids.keys()),
            "scores": mt.similarity_scores,
        })

    # Save sampled pairs for LLM validation
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "validation_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs_for_llm, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(pairs_for_llm)} validation pairs to {pairs_path}")

    # Try LLM-based validation if model is available
    try:
        _run_llm_validation(pairs_for_llm, cfg)
    except Exception as e:
        logger.warning(f"LLM validation skipped: {e}")
        print("\n⚠ LLM validation skipped. Review pairs manually above.")

    # Save matched results
    matched_path = output_dir / "vgbench_matched.json"
    with open(matched_path, "w") as f:
        json.dump([asdict(m) for m in matched], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved all {len(matched)} matches to {matched_path}")


def _run_llm_validation(pairs: list[dict], cfg) -> None:
    """Use local LLM to judge semantic consistency of matched pairs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = cfg.models.gen_model_7b
    logger.info(f"Loading LLM for validation: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )

    correct = 0
    total = len(pairs)

    print("\n" + "=" * 80)
    print("LLM SEMANTIC CONSISTENCY JUDGMENT")
    print("=" * 80)

    for i, pair in enumerate(pairs):
        descs = pair["descriptions"]
        if len(descs) < 2:
            continue

        prompt = f"""You are judging whether two task descriptions refer to the same visual concept.

Description A: {descs[0][:500]}

Description B: {descs[1][:500]}

Do these two descriptions describe the same visual concept or diagram? Answer ONLY "MATCH" or "NO_MATCH"."""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.0, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        is_match = "MATCH" in response.upper() and "NO_MATCH" not in response.upper()
        if is_match:
            correct += 1

        verdict = "✓ MATCH" if is_match else "✗ NO_MATCH"
        print(f"  Pair {i+1}: {verdict} (LLM: {response[:30]})")

    print(f"\n{'=' * 80}")
    print(f"RESULT: {correct}/{total} pairs judged as MATCH ({correct/total*100:.0f}%)")
    threshold_met = correct >= 16  # ≥16/20 = 80%
    print(f"Quality threshold (≥16/20): {'✓ PASSED' if threshold_met else '✗ FAILED'}")
    if threshold_met:
        print("→ MiniLM quality acceptable. Safe to lower threshold to 0.70.")
    else:
        print("→ MiniLM quality insufficient. Consider switching embedding model.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MiniLM matching quality")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override similarity threshold")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.threshold, args.n_samples, args.seed)
