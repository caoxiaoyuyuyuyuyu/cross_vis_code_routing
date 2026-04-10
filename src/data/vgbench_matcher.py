"""VGBench cross-format semantic matching (SVG + TikZ only).

Identifies semantically similar tasks across VGBench's SVG and TikZ
subsets, extracts format-neutral task descriptions, and builds matched
pairs for cross-format evaluation. Graphviz dropped per D002.

Data source: HuggingFace `vgbench/VGen` (5843 tasks)
Key fields: vformat (svg/tikz), caption (text description), code, idx
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.categories import to_mega_category

logger = logging.getLogger(__name__)


@dataclass
class MatchedTask:
    """A format-neutral task matched across VGBench subsets."""
    task_id: str
    neutral_description: str
    category: str
    source_ids: dict[str, str]  # format -> original VGBench idx
    eligible_formats: list[str]
    similarity_scores: dict[str, float] = field(default_factory=dict)  # "svg-tikz" -> score


# Category keywords for VGBench tasks (diagrams, geometric figures, etc.)
CATEGORY_KEYWORDS = {
    "flowchart": ["flowchart", "flow chart", "workflow", "process diagram",
                  "decision tree", "state machine", "state diagram",
                  "activity diagram", "pipeline", "process flow"],
    "graph": ["graph", "network", "node", "edge", "directed", "undirected",
              "tree structure", "binary tree", "linked list", "adjacency",
              "vertex", "vertices", "spanning tree", "dag", "dependency graph"],
    "sequence": ["sequence diagram", "timeline", "gantt", "schedule",
                 "message sequence", "interaction diagram", "swim lane"],
    "geometric": ["circle", "rectangle", "triangle", "polygon", "shape",
                  "geometric", "curve", "arc", "ellipse", "square",
                  "hexagon", "pentagon", "octagon", "parallelogram",
                  "trapezoid", "rhombus", "star shape", "arrow",
                  "coordinate", "angle", "perpendicular", "parallel lines",
                  "intersection", "tangent", "radius", "diameter"],
    "chart": ["bar chart", "pie chart", "histogram", "line chart",
              "scatter plot", "area chart", "plot", "axis", "x-axis",
              "y-axis", "data point", "legend", "bar graph"],
    "diagram": ["diagram", "architecture", "system diagram", "component",
                "class diagram", "er diagram", "uml", "entity relationship",
                "block diagram", "circuit", "schematic", "hierarchy",
                "organizational chart", "org chart", "mind map",
                "venn diagram", "concept map"],
    "icon": ["icon", "logo", "symbol", "badge", "emblem", "emoji",
             "pictogram", "glyph"],
    "illustration": ["illustration", "scene", "picture", "drawing",
                     "art", "landscape", "portrait", "cartoon",
                     "infographic", "visual", "figure", "image"],
    "table": ["table", "grid", "matrix", "spreadsheet", "tabular",
              "calendar", "timetable"],
    "map": ["map", "floor plan", "layout", "blueprint", "site plan",
            "geographic", "topology"],
}


def _classify_category(text: str) -> str:
    """Classify task into mega-category based on description keywords.

    Uses weighted scoring: longer keyword matches get higher weight
    to prefer specific matches over generic ones. Returns mega-category.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            if kw in text_lower:
                score += len(kw.split())
        scores[cat] = score
    best = max(scores, key=scores.get)
    fine = best if scores[best] > 0 else "other"
    return to_mega_category(fine)


class VGBenchMatcher:
    """Cross-format semantic matching for VGBench tasks.

    Uses sentence embeddings to find semantically similar tasks
    across SVG and TikZ subsets of VGBench (Graphviz dropped per D002).

    Args:
        vgbench_dir: Path to local VGBench data (or HuggingFace cache).
        embed_model: Model name for computing task embeddings.
        similarity_threshold: Minimum cosine similarity for a match.
        device: Device for embedding computation.
    """

    def __init__(
        self,
        vgbench_dir: str | Path,
        embed_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        similarity_threshold: float = 0.85,
        device: str = "cuda",
    ):
        self.vgbench_dir = Path(vgbench_dir)
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._tokenizer = None
        self._model = None

    def _init_embed_model(self) -> None:
        """Load embedding model lazily."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading embedding model: {self.embed_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.embed_model, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.embed_model, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

    def load_tasks(self) -> dict[str, list[dict[str, Any]]]:
        """Load VGBench VGen tasks grouped by format.

        Tries HuggingFace datasets first, falls back to local files.

        Returns:
            Dict mapping format name to list of task dicts with
            keys: idx, caption, code, vformat.
        """
        tasks_by_format: dict[str, list[dict[str, Any]]] = {
            "svg": [], "tikz": []
        }

        # Try loading from HuggingFace
        try:
            from datasets import load_dataset
            logger.info("Loading VGBench VGen from HuggingFace: vgbench/VGen")
            # vgbench/VGen exposes a single 'train' split with a 'vformat' column
            # (one of svg/tikz/graphviz); there is no 'test' split.
            ds = load_dataset("vgbench/VGen", split="train")
            for item in ds:
                fmt = item.get("vformat", "").lower().strip()
                if fmt in tasks_by_format:
                    tasks_by_format[fmt].append({
                        "idx": str(item.get("idx", "")),
                        "caption": item.get("caption", ""),
                        "code": item.get("code", ""),
                        "vformat": fmt,
                    })
            logger.info(
                f"Loaded from HF: "
                + ", ".join(f"{k}={len(v)}" for k, v in tasks_by_format.items())
            )
            return tasks_by_format
        except Exception as e:
            logger.warning(f"HuggingFace load failed ({e}), trying local files")

        # Fallback: local JSON files
        for fmt in tasks_by_format:
            fmt_dir = self.vgbench_dir / fmt
            if not fmt_dir.exists():
                logger.warning(f"Directory not found: {fmt_dir}")
                continue
            import json
            for json_file in sorted(fmt_dir.glob("*.json")):
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item["vformat"] = fmt
                        tasks_by_format[fmt].append(item)
                elif isinstance(data, dict):
                    for key, item in data.items():
                        if isinstance(item, dict):
                            item["vformat"] = fmt
                            item.setdefault("idx", key)
                            tasks_by_format[fmt].append(item)

        logger.info(
            f"Loaded locally: "
            + ", ".join(f"{k}={len(v)}" for k, v in tasks_by_format.items())
        )
        return tasks_by_format

    def _compute_embeddings_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """Compute mean-pooled embeddings for a list of texts.

        Args:
            texts: List of text strings.
            batch_size: Batch size for inference.

        Returns:
            Array of shape (len(texts), embed_dim).
        """
        self._init_embed_model()
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Mean pooling over non-padding tokens
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                token_embeds = outputs.last_hidden_state
                masked = token_embeds * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                mean_pooled = summed / counts

            # L2 normalize
            norms = mean_pooled.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = mean_pooled / norms
            all_embeddings.append(normalized.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _filter_valid_tasks(
        self, tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter tasks with non-empty captions."""
        filtered = [t for t in tasks if t.get("caption", "").strip()]
        logger.info(f"Filtered: {len(tasks)} -> {len(filtered)} (removed empty captions)")
        return filtered

    def match_across_formats(
        self,
        tasks_by_format: dict[str, list[dict[str, Any]]],
    ) -> list[MatchedTask]:
        """Find semantically matched tasks across formats.

        Strategy: for each pair of formats, compute cosine similarity
        between all task embeddings. For each task in the smaller set,
        find the best match in the larger set if above threshold.
        Then merge pairwise matches into groups.

        Args:
            tasks_by_format: Dict from format name to task list.

        Returns:
            List of MatchedTask instances.
        """
        # Filter and compute embeddings per format
        format_data: dict[str, tuple[list[dict], np.ndarray]] = {}
        for fmt, tasks in tasks_by_format.items():
            valid = self._filter_valid_tasks(tasks)
            if not valid:
                logger.warning(f"No valid tasks for format: {fmt}")
                continue
            captions = [t["caption"] for t in valid]
            embeddings = self._compute_embeddings_batch(captions)
            format_data[fmt] = (valid, embeddings)

        formats = list(format_data.keys())
        if len(formats) < 2:
            logger.error(f"Need at least 2 formats, got {len(formats)}")
            return []

        # Pairwise matching: build a graph of matched task indices
        # Key: (format, local_idx) -> set of matched (format, local_idx)
        match_graph: dict[tuple[str, int], dict[str, tuple[int, float]]] = {}

        for i, fmt_a in enumerate(formats):
            for fmt_b in formats[i + 1:]:
                tasks_a, emb_a = format_data[fmt_a]
                tasks_b, emb_b = format_data[fmt_b]

                # Cosine similarity matrix (already L2-normalized)
                sim_matrix = emb_a @ emb_b.T  # (n_a, n_b)

                # For each task in fmt_a, find best match in fmt_b
                for idx_a in range(len(tasks_a)):
                    best_b = int(sim_matrix[idx_a].argmax())
                    score = float(sim_matrix[idx_a, best_b])
                    if score >= self.similarity_threshold:
                        key_a = (fmt_a, idx_a)
                        match_graph.setdefault(key_a, {})[fmt_b] = (best_b, score)

                # For each task in fmt_b, find best match in fmt_a
                for idx_b in range(len(tasks_b)):
                    best_a = int(sim_matrix[:, idx_b].argmax())
                    score = float(sim_matrix[best_a, idx_b])
                    if score >= self.similarity_threshold:
                        key_b = (fmt_b, idx_b)
                        match_graph.setdefault(key_b, {})[fmt_a] = (best_a, score)

        # Build matched tasks from the graph
        matched: list[MatchedTask] = []
        used: set[tuple[str, int]] = set()
        task_counter = 0

        for (fmt, idx), matches in match_graph.items():
            if (fmt, idx) in used:
                continue

            # Collect all formats this task matches
            source_ids: dict[str, str] = {}
            eligible_formats: list[str] = []
            sim_scores: dict[str, float] = {}
            tasks_list, _ = format_data[fmt]

            source_ids[fmt] = tasks_list[idx]["idx"]
            eligible_formats.append(fmt)
            used.add((fmt, idx))

            for match_fmt, (match_idx, score) in matches.items():
                if (match_fmt, match_idx) not in used:
                    match_tasks, _ = format_data[match_fmt]
                    source_ids[match_fmt] = match_tasks[match_idx]["idx"]
                    eligible_formats.append(match_fmt)
                    sim_scores[f"{fmt}-{match_fmt}"] = score
                    used.add((match_fmt, match_idx))

            if len(eligible_formats) < 2:
                continue

            # Use the caption from the first format as neutral description
            caption = tasks_list[idx]["caption"]
            category = _classify_category(caption)

            matched.append(MatchedTask(
                task_id=f"vgb_{task_counter:04d}",
                neutral_description=caption,
                category=category,
                source_ids=source_ids,
                eligible_formats=sorted(eligible_formats),
                similarity_scores=sim_scores,
            ))
            task_counter += 1

        logger.info(f"Matched {len(matched)} cross-format tasks")

        # Log format coverage
        fmt_counts = {}
        for m in matched:
            for f in m.eligible_formats:
                fmt_counts[f] = fmt_counts.get(f, 0) + 1
        logger.info(f"Format coverage: {fmt_counts}")

        return matched

    def audit_cell_counts(
        self, matched: list[MatchedTask]
    ) -> dict[tuple[str, str], int]:
        """Count tasks per (category, format) cell.

        Args:
            matched: List of matched tasks.

        Returns:
            Dict mapping (category, format) to count.
        """
        counts: dict[tuple[str, str], int] = {}
        for task in matched:
            for fmt in task.eligible_formats:
                key = (task.category, fmt)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def run(self) -> list[MatchedTask]:
        """Full pipeline: load → embed → match → audit.

        Returns:
            List of matched, format-neutral tasks.
        """
        tasks_by_format = self.load_tasks()
        matched = self.match_across_formats(tasks_by_format)

        cell_counts = self.audit_cell_counts(matched)
        logger.info("VGBench cell counts (category × format):")
        for (cat, fmt), count in sorted(cell_counts.items()):
            logger.info(f"  {cat} × {fmt}: {count}")

        return matched

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
