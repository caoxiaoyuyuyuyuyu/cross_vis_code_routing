"""Phase 4: Routing + decoupling.

Steps:
1. Determine oracle best format per task from Phase 2/3 results
2. Train and evaluate VisRoute-Feature router
3. Train and evaluate VisRoute-Learned (BERT) router
4. Build and evaluate VisRoute-Proto router
5. Compare all routers against baselines (always-X, random, oracle)
6. Format decoupling analysis (two-pass vs single-pass)
"""

import argparse
import logging

from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = Config.from_yaml(config_path)

    # Step 1: Oracle labels
    logger.info("Step 1: Determining oracle best format per task")
    raise NotImplementedError("Phase 4 Step 1: Oracle labels")

    # Step 2: VisRoute-Feature
    logger.info("Step 2: VisRoute-Feature router")
    raise NotImplementedError("Phase 4 Step 2: VisRoute-Feature")

    # Step 3: VisRoute-Learned
    logger.info("Step 3: VisRoute-Learned (BERT) router")
    raise NotImplementedError("Phase 4 Step 3: VisRoute-Learned")

    # Step 4: VisRoute-Proto
    logger.info("Step 4: VisRoute-Proto router")
    raise NotImplementedError("Phase 4 Step 4: VisRoute-Proto")

    # Step 5: Comparison
    logger.info("Step 5: Router comparison")
    raise NotImplementedError("Phase 4 Step 5: Router comparison")

    # Step 6: Decoupling
    logger.info("Step 6: Format decoupling analysis")
    raise NotImplementedError("Phase 4 Step 6: Decoupling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Routing + decoupling")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
