"""FID (Fréchet Inception Distance) computation for visual quality."""

from pathlib import Path

import numpy as np


class FIDCalculator:
    """Computes FID between generated and reference image distributions.

    Uses InceptionV3 features following standard FID protocol.

    Args:
        batch_size: Batch size for feature extraction.
        device: Device for InceptionV3 inference.
    """

    def __init__(self, batch_size: int = 32, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _init_model(self) -> None:
        """Load InceptionV3 model lazily."""
        raise NotImplementedError

    def extract_features(self, image_dir: str | Path) -> np.ndarray:
        """Extract InceptionV3 features from all images in a directory.

        Args:
            image_dir: Directory containing PNG images.

        Returns:
            Feature array of shape (n_images, 2048).
        """
        raise NotImplementedError

    def compute_fid(
        self,
        generated_dir: str | Path,
        reference_dir: str | Path,
    ) -> float:
        """Compute FID between generated and reference image sets.

        Args:
            generated_dir: Directory of generated images.
            reference_dir: Directory of reference images.

        Returns:
            FID score (lower is better).
        """
        raise NotImplementedError

    def compute_fid_per_format(
        self,
        generated_dirs: dict[str, str | Path],
        reference_dirs: dict[str, str | Path],
    ) -> dict[str, float]:
        """Compute FID per format.

        Args:
            generated_dirs: Dict mapping format to generated image directory.
            reference_dirs: Dict mapping format to reference image directory.

        Returns:
            Dict mapping format to FID score.
        """
        raise NotImplementedError
