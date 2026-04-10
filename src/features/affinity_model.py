"""Logistic regression affinity model with feature ablation baselines.

Fits per-(model, format) logistic regressions predicting Execution Pass
from 5 structural features. Evaluates via held-out-category and
held-out-model generalization. Compares against distributional baselines.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AffinityResult:
    """Result of affinity model evaluation."""
    model_name: str  # generation model (7B/14B)
    format: str  # target format
    in_sample_auc: float
    cross_category_auc: float  # average over held-out folds
    per_fold_auc: dict[str, float]  # held-out category -> AUC
    coefficients: np.ndarray
    feature_names: list[str]
    bootstrap_ci: dict[str, tuple[float, float]]  # feature -> (lo, hi)


@dataclass
class AblationResult:
    """Result of feature ablation comparison."""
    structural_auc: float  # 5 structural features
    bow_auc: float  # BoW-5 baseline
    emb_pc_auc: float  # Emb-PC5 baseline
    random_auc: float  # Rand-5 baseline
    all_feat_auc: float  # structural + emb PCs


class AffinityModel:
    """Feature-grounded predictive model of format-task affinity.

    Core contribution of the paper. Fits logistic regressions
    and evaluates the structural-affinity claim.

    Args:
        n_bootstrap: Number of bootstrap iterations for CI estimation.
        seed: Random seed.
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        fmt: str,
    ) -> AffinityResult:
        """Fit logistic regression and compute in-sample AUC.

        Args:
            features: Feature matrix (n_tasks, 5).
            labels: Binary execution pass labels (n_tasks,).
            model_name: Generation model identifier.
            fmt: Target format.

        Returns:
            AffinityResult with in-sample metrics.
        """
        raise NotImplementedError

    def evaluate_held_out_category(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
        model_name: str,
        fmt: str,
    ) -> AffinityResult:
        """Leave-one-category-out cross-validation.

        Args:
            features: Feature matrix (n_tasks, 5).
            labels: Binary labels.
            categories: Category labels for each task.
            model_name: Generation model identifier.
            fmt: Target format.

        Returns:
            AffinityResult with cross-category AUC.
        """
        raise NotImplementedError

    def evaluate_held_out_model(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> float:
        """Train on one model size (7B), test on another (14B).

        Args:
            train_features: Features from training model.
            train_labels: Labels from training model.
            test_features: Features from test model.
            test_labels: Labels from test model.

        Returns:
            Transfer AUC.
        """
        raise NotImplementedError

    def bootstrap_attribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, tuple[float, float]]:
        """Bootstrap confidence intervals for feature coefficients.

        Args:
            features: Feature matrix.
            labels: Binary labels.
            feature_names: Names of features.

        Returns:
            Dict mapping feature name to (lower, upper) 95% CI.
        """
        raise NotImplementedError

    def run_ablation(
        self,
        structural_features: np.ndarray,
        bow_features: np.ndarray,
        emb_pc_features: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
    ) -> AblationResult:
        """Run feature ablation against distributional baselines.

        Compares 5 structural features against:
        - BoW-5: top-5 TF-IDF keywords
        - Emb-PC5: 5 PCs from frozen embeddings
        - Rand-5: random features (chance control)
        - All-feat: structural + emb PCs

        Args:
            structural_features: (n, 5) structural features.
            bow_features: (n, 5) BoW features.
            emb_pc_features: (n, 5) embedding PC features.
            labels: Binary labels.
            categories: Category labels.

        Returns:
            AblationResult with AUCs for each baseline.
        """
        raise NotImplementedError

    def check_preregistered_criterion(
        self, results: list[AffinityResult]
    ) -> dict[str, Any]:
        """Check pre-registered AUC ≥ 0.70 criterion.

        Determines tiered conclusion:
        T1: All 3 formats meet threshold → strong confirmation.
        T2: SVG+TikZ meet, Graphviz doesn't → partial confirmation.
        T3: SVG+TikZ also below → falsification.

        Args:
            results: AffinityResults for all (model, format) pairs.

        Returns:
            Dict with 'tier', 'explanation', per-format AUCs.
        """
        raise NotImplementedError
