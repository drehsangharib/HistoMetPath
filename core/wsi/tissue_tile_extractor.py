from pathlib import Path
import json

import numpy as np
import openslide

from core.wsi.tissue_mask import (
    create_tissue_mask,
)


TILE_SIZE = 256


def extract_tissue_tiles(
    slide_path,
    output_dir,
    stride=4096,
    max_tiles=200,
    tissue_fraction_threshold=0.25,
):
    slide = openslide.OpenSlide(
        str(slide_path)
    )

    width, height = slide.dimensions

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted = []
    rejected = 0

    count = 0

    for y in range(
        0,
        height,
        stride,
    ):
        for x in range(
            0,
            width,
            stride,
        ):

            if count >= max_tiles:
                break

            tile = slide.read_region(
                (x, y),
                0,
                (
                    TILE_SIZE,
                    TILE_SIZE,
                ),
            ).convert("RGB")

            mask = create_tissue_mask(
                tile
            )

            frac = float(
                mask.mean()
            )

            if (
                frac
                < tissue_fraction_threshold
            ):
                rejected += 1
                continue

            tile_name = (
                f"x{x}_y{y}.png"
            )

            tile.save(
                output_dir
                / tile_name
            )

            accepted.append(
                {
                    "tile_name": tile_name,
                    "x": x,
                    "y": y,
                    "tissue_fraction": frac,
                }
            )

            count += 1

        if count >= max_tiles:
            break

    slide.close()

    return accepted, rejected


def main():

    root = Path(__file__).resolve().parents[2]

    slides = [
        root
        / "data"
        / "camelyon16"
        / "training"
        / "normal"
        / "normal_100.tif",

        root
        / "data"
        / "camelyon16"
        / "training"
        / "tumor"
        / "tumor_100.tif",
    ]

    output_root = (
        root
        / "outputs"
        / "camelyon16"
        / "tissue_tiles"
    )

    manifest_root = (
        root
        / "outputs"
        / "camelyon16"
        / "tissue_manifests"
    )

    manifest_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for slide_path in slides:

        if not slide_path.exists():
            continue

        slide_name = (
            slide_path.stem
        )

        accepted, rejected = (
            extract_tissue_tiles(
                slide_path,
                output_root
                / slide_name,
            )
        )

        manifest_file = (
            manifest_root
            / f"{slide_name}.json"
        )

        manifest_file.write_text(
            json.dumps(
                accepted,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary.append(
            {
                "slide": slide_name,
                "accepted": len(
                    accepted
                ),
                "rejected": rejected,
            }
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
