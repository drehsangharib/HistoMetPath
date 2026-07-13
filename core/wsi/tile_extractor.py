from pathlib import Path
import json

import openslide

TILE_SIZE = 256


def extract_tiles(
    slide_path,
    output_dir,
    stride=4096,
    max_tiles=200,
):
    slide = openslide.OpenSlide(str(slide_path))

    width, height = slide.dimensions

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    count = 0

    for y in range(0, height, stride):
        for x in range(0, width, stride):

            if count >= max_tiles:
                break

            tile = slide.read_region(
                (x, y),
                0,
                (TILE_SIZE, TILE_SIZE),
            ).convert("RGB")

            tile_name = f"x{x}_y{y}.png"

            tile_path = (
                output_dir / tile_name
            )

            tile.save(tile_path)

            manifest.append(
                {
                    "tile_name": tile_name,
                    "x": x,
                    "y": y,
                }
            )

            count += 1

        if count >= max_tiles:
            break

    slide.close()

    return manifest


def main():

    root = Path(__file__).resolve().parents[2]

    slides = [
        root / "data" / "camelyon16" / "training" / "normal" / "normal_100.tif",
        root / "data" / "camelyon16" / "training" / "tumor" / "tumor_100.tif",
    ]

    out_root = (
        root
        / "outputs"
        / "camelyon16"
        / "tiles"
    )

    manifest_root = (
        root
        / "outputs"
        / "camelyon16"
        / "tile_manifests"
    )

    manifest_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for slide_path in slides:

        if not slide_path.exists():
            continue

        slide_name = slide_path.stem

        slide_output = (
            out_root
            / slide_name
        )

        manifest = extract_tiles(
            slide_path=slide_path,
            output_dir=slide_output,
        )

        manifest_path = (
            manifest_root
            / f"{slide_name}.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            slide_name,
            len(manifest),
        )


if __name__ == "__main__":
    main()
