"""Global configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    gen_model_7b: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    gen_model_14b: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    cross_model: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    judge_model_primary: str = "OpenGVLab/InternVL3-8B"
    judge_model_secondary: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    inter_judge_sample_size: int = 300
    embed_model: str = "Qwen/Qwen2.5-7B-Instruct"
    router_backbone: str = "bert-base-uncased"


@dataclass
class DataConfig:
    vgbench_dir: str = "data/vgbench"
    visplotbench_dir: str = "data/visplotbench"
    output_dir: str = "data/processed"
    formats: list[str] = field(default_factory=lambda: ["svg", "tikz", "graphviz"])
    min_cell_count: int = 50
    similarity_threshold: float = 0.85


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 8
    num_samples: int = 1
    vllm_tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.90


@dataclass
class RenderConfig:
    svg_timeout: int = 10
    tikz_timeout: int = 30
    graphviz_timeout: int = 10
    tikz_max_workers: int = 4
    output_dpi: int = 300


@dataclass
class EvalConfig:
    fid_batch_size: int = 32
    judge_batch_size: int = 4
    inter_judge_sample_size: int = 200
    kappa_threshold: float = 0.6


@dataclass
class FeatureConfig:
    extractor_a_model: str = "Qwen/Qwen2.5-7B-Instruct"
    extractor_b_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    agreement_pearson_threshold: float = 0.75
    agreement_kappa_threshold: float = 0.6
    n_bootstrap: int = 1000


@dataclass
class RoutingConfig:
    bert_lr: float = 2e-5
    bert_epochs: int = 5
    bert_batch_size: int = 32
    proto_k: int = 5


@dataclass
class DecouplingConfig:
    pseudocode_max_tokens: int = 1024
    conversion_max_tokens: int = 2048


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    decoupling: DecouplingConfig = field(default_factory=DecouplingConfig)
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file, overriding defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "Config":
        cfg = cls()
        section_map = {
            "models": (cfg.models, ModelConfig),
            "data": (cfg.data, DataConfig),
            "generation": (cfg.generation, GenerationConfig),
            "render": (cfg.render, RenderConfig),
            "evaluation": (cfg.evaluation, EvalConfig),
            "features": (cfg.features, FeatureConfig),
            "routing": (cfg.routing, RoutingConfig),
            "decoupling": (cfg.decoupling, DecouplingConfig),
        }
        for key, val in d.items():
            if key in section_map and isinstance(val, dict):
                obj, _ = section_map[key]
                for k, v in val.items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg
