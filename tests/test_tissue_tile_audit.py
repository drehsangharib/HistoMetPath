import numpy as np
from PIL import Image

from core.wsi.tissue_mask import (
    create_tissue_mask,
    tissue_fraction,
)


def test_half_tissue_tile_fraction():
    array = np.full(
        (32, 32, 3),
        255,
        dtype=np.uint8,
    )

    array[:, :16, :] = 80

    image = Image.fromarray(array)

    mask = create_tissue_mask(image)
    fraction = tissue_fraction(mask)

    assert mask.shape == (32, 32)
    assert 0.49 <= fraction <= 0.51
