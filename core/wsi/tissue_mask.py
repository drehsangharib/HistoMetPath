"""Simple and reproducible tissue-mask utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image


def create_tissue_mask(
    image: Image.Image,
    intensity_threshold: int = 220,
) -> np.ndarray:
    """Return a Boolean mask where darker pixels represent tissue."""

    if not 0 <= intensity_threshold <= 255:
        raise ValueError(
            "intensity_threshold must be between 0 and 255."
        )

    gray = image.convert("L")

    array = np.asarray(
        gray,
        dtype=np.uint8,
    )

    return array < intensity_threshold


def tissue_fraction(mask: np.ndarray) -> float:
    """Return the fraction of pixels classified as tissue."""

    mask_array = np.asarray(
        mask,
        dtype=bool,
    )

    if mask_array.size == 0:
        raise ValueError("Tissue mask must not be empty.")

    return float(mask_array.mean())
