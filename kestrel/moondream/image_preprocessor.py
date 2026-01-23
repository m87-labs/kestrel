"""Image preprocessing with optimized parallel execution.

This module provides a thread pool for image preprocessing operations.
"""

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pyvips

from kestrel.moondream.config import VisionConfig
from kestrel.moondream.image_crops import OverlapCropOutput, overlap_crop_image
from kestrel.utils.image import ensure_srgb


class ImagePreprocessor:
    """Manages threaded image preprocessing.

    This class encapsulates the thread pool needed for efficient parallel
    image preprocessing.

    Usage:
        preprocessor = ImagePreprocessor()
        future = preprocessor.submit(image, vision_config)
        result = future.result()  # OverlapCropOutput

        # Or for synchronous processing:
        result = preprocessor.preprocess(image, vision_config)

        # Shutdown when done:
        preprocessor.shutdown()
    """

    def __init__(self, num_workers: int = 16):
        """Initialize the image preprocessor.

        Args:
            num_workers: Number of Python threads for parallel preprocessing.
        """
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="kestrel-img",
        )

    def preprocess(
        self,
        image: pyvips.Image | np.ndarray,
        config: VisionConfig,
    ) -> OverlapCropOutput:
        """Preprocess an image synchronously.

        Normalizes the image to sRGB and creates overlap crops for the
        vision encoder.

        Args:
            image: Input image as pyvips.Image or numpy array.
            config: Vision configuration with crop parameters.

        Returns:
            OverlapCropOutput with crops array and tiling info.
        """
        normalized = ensure_srgb(image)
        return overlap_crop_image(
            normalized,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

    def submit(
        self,
        image: pyvips.Image | np.ndarray,
        config: VisionConfig,
    ) -> Future[OverlapCropOutput]:
        """Submit an image for asynchronous preprocessing.

        Args:
            image: Input image as pyvips.Image or numpy array.
            config: Vision configuration with crop parameters.

        Returns:
            Future that resolves to OverlapCropOutput.
        """
        return self._executor.submit(self.preprocess, image, config)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        self._executor.shutdown(wait=wait)
