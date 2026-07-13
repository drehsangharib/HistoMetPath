import numpy as np
from PIL import Image

from core.wsi.tissue_mask import (
    create_tissue_mask,
)


def test_tissue_mask():

    img = Image.new(
        "RGB",
        (100, 100),
        "white",
    )

    mask = create_tissue_mask(
        img
    )

    assert (
        np.mean(mask)
        < 0.01
    )
