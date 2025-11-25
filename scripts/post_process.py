import cv2
import numpy as np
from typing import Tuple


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str = "holes"
) -> Tuple[np.ndarray, bool]:
    """
    Remove small disconnected regions or holes in a binary mask.
    """
    assert mode in ["holes", "islands"], "mode must be 'holes' or 'islands'"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask.astype(bool), False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    cleaned = np.isin(regions, fill_labels)
    return cleaned.astype(bool), True


def morph_close(mask: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = np.ones((k, k), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(bool)


def clean_refined_mask(mask: np.ndarray, area_frac: float = 0.0025) -> np.ndarray:
    h, w = mask.shape
    area_thresh = max(1.0, area_frac * h * w)
    mask_bool = mask.astype(bool)
    mask_bool, _ = remove_small_regions(mask_bool, area_thresh, mode="holes")
    mask_bool, _ = remove_small_regions(mask_bool, area_thresh, mode="islands")
    mask_bool = morph_close(mask_bool, k=3)
    return mask_bool
