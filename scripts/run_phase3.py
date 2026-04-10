"""Phase 3: Evaluation + features + affinity modeling.

Steps:
1. VL-judge evaluation (7B + 32B inter-judge agreement on 200-sample slice)
2. FID computation per format
3. Dual-extractor feature extraction + agreement check
4. Affinity logistic regression + held-out evaluation
5. Feature ablation (BoW-5, Emb-PC5, Rand-5, All-feat)
6. Failure-mode taxonomy
"""

import argparse
import logging

from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = Config.from_yaml(config_path)

    # Step 1: VL-judge evaluation
    logger.info("Step 1: Visual faithfulness evaluation")
    # TODO: Load render results, run VL-7B judge on all, VL-32B on 200-sample slice
    raise NotImplementedError("Phase 3 Step 1: VL-judge evaluation")

    # Step 2: FID
    logger.info("Step 2: FID computation")
    raise NotImplementedError("Phase 3 Step 2: FID computation")

    # Step 3: Feature extraction
    logger.info("Step 3: Dual-extractor feature extraction")
    raise NotImplementedError("Phase 3 Step 3: Feature extraction")

    # Step 4: Affinity modeling
    logger.info("Step 4: Affinity logistic regression")
    raise NotImplementedError("Phase 3 Step 4: Affinity modeling")

    # Step 5: Feature ablation
    logger.info("Step 5: Feature ablation baselines")
    raise NotImplementedError("Phase 3 Step 5: Feature ablation")

    # Step 6: Failure taxonomy
    logger.info("Step 6: Failure-mode taxonomy")
    raise NotImplementedError("Phase 3 Step 6: Failure taxonomy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Eval + features + affinity")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
