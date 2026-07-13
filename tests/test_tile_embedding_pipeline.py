import json

import numpy as np
from PIL import Image

from core.wsi.extract_tile_embeddings import ManifestTileDataset, build_transform


def test_manifest_tile_dataset(tmp_path):
    tile_directory = tmp_path / "tiles"
    tile_directory.mkdir()
    image_path = tile_directory / "x10_y20.png"
    Image.new("RGB", (256, 256), (120, 80, 140)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "tile_name": image_path.name,
                    "x": 10,
                    "y": 20,
                    "tissue_fraction": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = ManifestTileDataset(
        tile_directory,
        manifest_path,
        build_transform(96),
    )
    tensor, coordinates, tile_name = dataset[0]
    assert tuple(tensor.shape) == (3, 96, 96)
    assert np.array_equal(coordinates.numpy(), np.array([10, 20]))
    assert tile_name == "x10_y20.png"
