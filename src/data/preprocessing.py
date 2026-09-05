"""Small, dataset-agnostic preprocessing helpers."""

import numpy as np


def minmax_normalize(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32)
    lo = float(image.min())
    hi = float(image.max())
    return (image - lo) / (hi - lo + eps)


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask >= threshold).astype(np.float32)
