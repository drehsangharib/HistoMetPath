from pathlib import Path

import numpy as np
from PIL import Image


def create_tissue_mask(
    thumbnail,
    brightness_threshold=220,
):
    """
    Detect likely tissue using RGB brightness.

    White background is removed.
    """

    image = np.asarray(
        thumbnail.convert("RGB")
    )

    gray = image.mean(
        axis=2
    )

    mask = (
        gray
        < brightness_threshold
    )

    return mask


def tissue_fraction(mask):
    return float(mask.mean())


if __name__ == "__main__":
    img = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    mask = create_tissue_mask(img)

    print(
        tissue_fraction(mask)
    )
