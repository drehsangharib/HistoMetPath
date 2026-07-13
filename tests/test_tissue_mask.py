import numpy as np
from PIL import Image

from core.wsi.tissue_mask import (
    create_tissue_mask,
    tissue_fraction,
)


def test_white_tile_is_background():
    image = Image.new(
        "RGB",
        (32, 32),
        (255, 255, 255),
    )

    mask = create_tissue_mask(image)

    assert mask.shape == (32, 32)
    assert tissue_fraction(mask) == 0.0


def test_dark_tile_is_tissue():
    image = Image.new(
        "RGB",
        (32, 32),
        (80, 40, 90),
    )

    mask = create_tissue_mask(image)

    assert mask.shape == (32, 32)
    assert tissue_fraction(mask) == 1.0


def test_intensity_threshold_is_configurable():
    array = np.full(
        (16, 16, 3),
        180,
        dtype=np.uint8,
    )

    image = Image.fromarray(array)

    low_threshold_mask = create_tissue_mask(
        image,
        intensity_threshold=170,
    )

    high_threshold_mask = create_tissue_mask(
        image,
        intensity_threshold=220,
    )

    assert tissue_fraction(low_threshold_mask) == 0.0
    assert tissue_fraction(high_threshold_mask) == 1.0


def test_invalid_threshold_raises_value_error():
    image = Image.new(
        "RGB",
        (8, 8),
        (100, 100, 100),
    )

    try:
        create_tissue_mask(
            image,
            intensity_threshold=300,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid threshold."
        )
