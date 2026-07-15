"""Audit whether sampled CAMELYON16 tumor-slide tiles intersect XML lesions.

This is a post-test diagnostic. It does not generate model probabilities,
change thresholds, retrain models, or alter the completed final-test result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit annotation coverage of sampled tumor-slide tiles."
    )
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_lesion_coverage_audit.yaml",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    required = {
        "processing_manifest",
        "embedding_root",
        "annotation_root",
        "final_test_result",
        "final_test_lock",
        "output_root",
        "tile_size",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    if config.get("prohibit_model_outputs") is not True:
        raise RuntimeError("Audit config must prohibit model outputs.")
    return config


def parse_polygons(xml_path: Path) -> list[np.ndarray]:
    tree = ET.parse(xml_path)
    polygons: list[np.ndarray] = []

    for annotation in tree.findall(".//Annotation"):
        points = []
        coordinates = annotation.findall(".//Coordinate")
        coordinates = sorted(
            coordinates,
            key=lambda node: float(node.attrib.get("Order", 0)),
        )
        for coordinate in coordinates:
            x_value = coordinate.attrib.get("X")
            y_value = coordinate.attrib.get("Y")
            if x_value is None or y_value is None:
                continue
            points.append((float(x_value), float(y_value)))
        if len(points) >= 3:
            array = np.asarray(points, dtype=np.float64)
            if not np.array_equal(array[0], array[-1]):
                array = np.vstack([array, array[0]])
            polygons.append(array)

    if not polygons:
        raise RuntimeError(f"No valid lesion polygons found in {xml_path}")
    return polygons


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    for first, second in zip(polygon[:-1], polygon[1:]):
        x1, y1 = first
        x2, y2 = second
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def orientation(a, b, c) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if math.isclose(value, 0.0, abs_tol=1e-9):
        return 0
    return 1 if value > 0 else 2


def on_segment(a, b, c) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(a, b, c, d) -> bool:
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def rectangle_intersects_polygon(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    polygon: np.ndarray,
) -> bool:
    polygon_x_min = float(polygon[:, 0].min())
    polygon_y_min = float(polygon[:, 1].min())
    polygon_x_max = float(polygon[:, 0].max())
    polygon_y_max = float(polygon[:, 1].max())
    if (
        x_max < polygon_x_min
        or polygon_x_max < x_min
        or y_max < polygon_y_min
        or polygon_y_max < y_min
    ):
        return False

    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    if any(point_in_polygon(x, y, polygon) for x, y in corners):
        return True
    if any(
        x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
        for point in polygon[:-1]
    ):
        return True

    rectangle_edges = list(zip(corners, corners[1:] + corners[:1]))
    polygon_edges = list(zip(map(tuple, polygon[:-1]), map(tuple, polygon[1:])))
    return any(
        segments_intersect(rect_start, rect_end, poly_start, poly_end)
        for rect_start, rect_end in rectangle_edges
        for poly_start, poly_end in polygon_edges
    )


def audit_slide(
    row: dict,
    coordinates_path: Path,
    xml_path: Path,
    tile_size: int,
) -> dict:
    coordinates = np.load(coordinates_path, allow_pickle=False)
    polygons = parse_polygons(xml_path)
    footprint = float(tile_size) * float(row["selected_downsample"])

    hit_tile_indices: list[int] = []
    polygon_hits = [0 for _ in polygons]
    for tile_index, (x_value, y_value) in enumerate(coordinates):
        x_min, y_min = float(x_value), float(y_value)
        x_max, y_max = x_min + footprint, y_min + footprint
        hit_polygons = [
            polygon_index
            for polygon_index, polygon in enumerate(polygons)
            if rectangle_intersects_polygon(x_min, y_min, x_max, y_max, polygon)
        ]
        if hit_polygons:
            hit_tile_indices.append(tile_index)
            for polygon_index in hit_polygons:
                polygon_hits[polygon_index] += 1

    tile_count = int(len(coordinates))
    hit_count = int(len(hit_tile_indices))
    covered_polygons = sum(value > 0 for value in polygon_hits)
    return {
        "slide": row["slide"],
        "split": row["split"],
        "label": row["label"],
        "tile_count": tile_count,
        "tile_footprint_level_zero_pixels": footprint,
        "annotation_polygon_count": len(polygons),
        "lesion_intersecting_tile_count": hit_count,
        "lesion_intersecting_tile_fraction": hit_count / tile_count,
        "covered_polygon_count": int(covered_polygons),
        "covered_polygon_fraction": covered_polygons / len(polygons),
        "bag_contains_annotated_lesion": hit_count > 0,
        "lesion_intersecting_tile_indices": hit_tile_indices,
        "annotation_sha256": sha256_file(xml_path),
        "coordinates_sha256": sha256_file(coordinates_path),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "lesion_intersecting_tile_indices"
            }
        )
    fieldnames = list(output_rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    final_result_path = project_path(config["final_test_result"])
    final_lock_path = project_path(config["final_test_lock"])
    if not final_result_path.is_file() or not final_lock_path.is_file():
        raise RuntimeError("Completed one-time final-test evidence is required.")

    manifest_path = project_path(config["processing_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding_root = project_path(config["embedding_root"])
    annotation_root = project_path(config["annotation_root"])
    included_splits = set(config.get("include_tumor_splits", ["train", "validation", "test"]))

    tumor_rows = [
        row
        for row in manifest["slides"]
        if row["label"] == "tumor" and row["split"] in included_splits
    ]
    tumor_rows.sort(key=lambda row: row["slide"])
    if len(tumor_rows) != 21:
        raise RuntimeError(f"Expected 21 tumor slides; found {len(tumor_rows)}")

    results = []
    for row in tumor_rows:
        coordinates_path = embedding_root / f"{row['slide']}_coordinates.npy"
        xml_path = annotation_root / f"{row['slide']}.xml"
        if not coordinates_path.is_file():
            raise FileNotFoundError(coordinates_path)
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        print(f"Auditing {row['slide']} ({row['split']})")
        results.append(
            audit_slide(row, coordinates_path, xml_path, int(config["tile_size"]))
        )

    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    uncovered = [row["slide"] for row in results if not row["bag_contains_annotated_lesion"]]
    test_results = [row for row in results if row["split"] == "test"]
    output = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "scientific_scope": config["scientific_scope"],
        "model_outputs_generated": False,
        "final_test_result_sha256": sha256_file(final_result_path),
        "final_test_lock_sha256": sha256_file(final_lock_path),
        "processing_manifest_sha256": sha256_file(manifest_path),
        "tumor_slide_count": len(results),
        "tumor_bags_with_annotated_lesion": sum(
            row["bag_contains_annotated_lesion"] for row in results
        ),
        "tumor_bags_without_annotated_lesion": len(uncovered),
        "uncovered_tumor_slides": uncovered,
        "fresh_test_tumor_results": test_results,
        "slides": results,
        "limitations": [
            "Post-test diagnostic only; the completed final-test result is unchanged.",
            "Polygon intersection establishes sampled lesion coverage, not biological adequacy.",
            "No model probabilities, threshold changes, or retraining are produced.",
        ],
        "passed": True,
    }
    json_path = output_root / "lesion_coverage_audit.json"
    csv_path = output_root / "lesion_coverage_summary.csv"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(csv_path, results)
    print()
    print(json.dumps({
        "tumor_slide_count": output["tumor_slide_count"],
        "tumor_bags_with_annotated_lesion": output["tumor_bags_with_annotated_lesion"],
        "tumor_bags_without_annotated_lesion": output["tumor_bags_without_annotated_lesion"],
        "uncovered_tumor_slides": output["uncovered_tumor_slides"],
        "model_outputs_generated": False,
        "passed": True,
    }, indent=2))
    print(f"Audit written to: {json_path}")
    print("PASS: Annotation-aware lesion-coverage audit completed without model evaluation.")


if __name__ == "__main__":
    main()
