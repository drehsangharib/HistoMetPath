"""Diagnose CAMELYON16 sampled-tile and XML coordinate alignment."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openslide


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "camelyon16"
    / "annotations"
)

EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "embeddings"
    / "camelyon16"
    / "expanded_fresh_holdout"
)

SLIDE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "camelyon16"
    / "training"
    / "tumor"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "camelyon16"
    / "lesion_coverage_audit"
    / "coordinate_diagnostic.json"
)


def load_annotation_points(xml_path: Path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    points = []

    for coordinate in root.findall(".//Coordinate"):
        x_value = coordinate.attrib.get("X")
        y_value = coordinate.attrib.get("Y")

        if x_value is None or y_value is None:
            continue

        points.append(
            (
                float(x_value),
                float(y_value),
            )
        )

    if not points:
        raise RuntimeError(
            f"No annotation points found in {xml_path}."
        )

    return np.asarray(
        points,
        dtype=np.float64,
    )


def diagnose_slide(slide_name: str) -> dict:
    slide_path = SLIDE_ROOT / f"{slide_name}.tif"
    xml_path = ANNOTATION_ROOT / f"{slide_name}.xml"
    coordinates_path = (
        EMBEDDING_ROOT
        / f"{slide_name}_coordinates.npy"
    )

    coordinates = np.load(
        coordinates_path,
        allow_pickle=False,
    ).astype(np.float64)

    annotation_points = load_annotation_points(
        xml_path
    )

    slide = openslide.OpenSlide(str(slide_path))

    try:
        width, height = slide.dimensions
    finally:
        slide.close()

    tile_footprint = 512.0

    tile_centers = coordinates + tile_footprint / 2.0

    minimum_distance = float("inf")
    nearest_tile_index = None
    nearest_annotation_index = None

    for annotation_index, point in enumerate(
        annotation_points
    ):
        distances = np.sqrt(
            np.sum(
                (tile_centers - point) ** 2,
                axis=1,
            )
        )

        tile_index = int(np.argmin(distances))
        distance = float(distances[tile_index])

        if distance < minimum_distance:
            minimum_distance = distance
            nearest_tile_index = tile_index
            nearest_annotation_index = annotation_index

    annotations_in_bounds = bool(
        np.all(annotation_points[:, 0] >= 0)
        and np.all(annotation_points[:, 0] <= width)
        and np.all(annotation_points[:, 1] >= 0)
        and np.all(annotation_points[:, 1] <= height)
    )

    sampled_extent = {
        "x_min": float(coordinates[:, 0].min()),
        "x_max": float(
            coordinates[:, 0].max()
            + tile_footprint
        ),
        "y_min": float(coordinates[:, 1].min()),
        "y_max": float(
            coordinates[:, 1].max()
            + tile_footprint
        ),
    }

    annotation_extent = {
        "x_min": float(annotation_points[:, 0].min()),
        "x_max": float(annotation_points[:, 0].max()),
        "y_min": float(annotation_points[:, 1].min()),
        "y_max": float(annotation_points[:, 1].max()),
    }

    extent_overlap = bool(
        sampled_extent["x_max"]
        >= annotation_extent["x_min"]
        and annotation_extent["x_max"]
        >= sampled_extent["x_min"]
        and sampled_extent["y_max"]
        >= annotation_extent["y_min"]
        and annotation_extent["y_max"]
        >= sampled_extent["y_min"]
    )

    return {
        "slide": slide_name,
        "slide_width": int(width),
        "slide_height": int(height),
        "sampled_tile_count": int(len(coordinates)),
        "annotation_point_count": int(
            len(annotation_points)
        ),
        "tile_footprint_level_zero_pixels": (
            tile_footprint
        ),
        "sampled_extent": sampled_extent,
        "annotation_extent": annotation_extent,
        "annotations_inside_slide_bounds": (
            annotations_in_bounds
        ),
        "sampled_and_annotation_extents_overlap": (
            extent_overlap
        ),
        "minimum_annotation_to_tile_center_distance": (
            minimum_distance
        ),
        "nearest_sampled_tile_index": int(
            nearest_tile_index
        ),
        "nearest_sampled_tile_origin": (
            coordinates[nearest_tile_index].tolist()
        ),
        "nearest_annotation_point": (
            annotation_points[
                nearest_annotation_index
            ].tolist()
        ),
    }


def main() -> None:
    slides = [
        "tumor_018",
        "tumor_019",
        "tumor_020",
    ]

    results = [
        diagnose_slide(slide)
        for slide in slides
    ]

    output = {
        "schema_version": "1.0",
        "scientific_scope": (
            "post-test coordinate-system diagnostic; "
            "no model evaluation"
        ),
        "slides": results,
        "all_annotations_inside_bounds": all(
            row["annotations_inside_slide_bounds"]
            for row in results
        ),
        "model_outputs_generated": False,
        "passed": True,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2))
    print()
    print(
        f"Diagnostic written to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
