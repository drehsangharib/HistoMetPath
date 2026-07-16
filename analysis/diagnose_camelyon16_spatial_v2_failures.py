"""Diagnose Spatial Sampler v2 lesion misses on development slides only.

Annotations are used only after the frozen v2 sampler has completed. The script
rescans the candidate grid to determine whether lesion-intersecting candidates
existed, passed the tissue threshold, which bins contained them, how much v2
allocation those bins received, and how close selected coordinates came to the
lesions. No embeddings, model probabilities, thresholds, or test slides are
loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openslide
import yaml

from analysis.audit_camelyon16_lesion_coverage import parse_polygons, rectangle_intersects_polygon
from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler import spatial_bin
from core.wsi.tissue_mask import create_tissue_mask, tissue_fraction


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose development-only Spatial v2 misses")
    parser.add_argument("--config", default="configs/wsi/camelyon16_spatial_v2_failure_analysis.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    required = {
        "processing_manifest", "spatial_v2_manifest", "spatial_v2_coordinate_root",
        "annotation_root", "output_root", "tile_size", "stride",
        "intensity_threshold", "min_tissue_fraction", "spatial_grid_rows",
        "spatial_grid_columns", "allowed_splits", "prohibited_splits",
        "expected_uncovered_slides",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    if "test" not in config["prohibited_splits"]:
        raise RuntimeError("Test split must remain prohibited")
    return config


def tile_hits_any_polygon(x: int, y: int, footprint: float, polygons: list[np.ndarray]) -> bool:
    return any(
        rectangle_intersects_polygon(x, y, x + footprint, y + footprint, polygon)
        for polygon in polygons
    )


def minimum_center_distance(selected: np.ndarray, polygons: list[np.ndarray], footprint: float) -> float:
    centers = selected.astype(np.float64) + footprint / 2.0
    points = np.concatenate([polygon[:-1] for polygon in polygons], axis=0)
    minimum = math.inf
    for point in points:
        distances = np.sqrt(np.sum((centers - point) ** 2, axis=1))
        minimum = min(minimum, float(distances.min()))
    return minimum


def categorize(result: dict) -> str:
    if result["lesion_grid_candidate_count"] == 0:
        return "grid_resolution_or_lesion_geometry_miss"
    if result["lesion_tissue_eligible_candidate_count"] == 0:
        return "tissue_threshold_exclusion"
    if result["lesion_bin_selected_tile_count"] == 0:
        return "allocation_exclusion"
    return "within_bin_selection_miss"


def diagnose_slide(row: dict, v2_row: dict, selected_path: Path, xml_path: Path, config: dict) -> dict:
    polygons = parse_polygons(xml_path)
    slide = openslide.OpenSlide(str(Path(row["path"])))
    try:
        level = int(row["selected_level"])
        downsample = float(row["selected_downsample"])
        level_width, level_height = slide.level_dimensions[level]
        slide_width, slide_height = slide.dimensions
        tile_size = int(config["tile_size"])
        stride = int(config["stride"])
        footprint = tile_size * downsample
        grid_rows = int(config["spatial_grid_rows"])
        grid_columns = int(config["spatial_grid_columns"])
        lesion_grid_candidates = 0
        lesion_tissue_eligible = 0
        lesion_eligible_fractions: list[float] = []
        lesion_bins: set[tuple[int, int]] = set()
        candidates_examined = 0

        for y_level in range(0, level_height - tile_size + 1, stride):
            for x_level in range(0, level_width - tile_size + 1, stride):
                candidates_examined += 1
                x_zero = int(round(x_level * downsample))
                y_zero = int(round(y_level * downsample))
                if not tile_hits_any_polygon(x_zero, y_zero, footprint, polygons):
                    continue
                lesion_grid_candidates += 1
                tile = slide.read_region((x_zero, y_zero), level, (tile_size, tile_size)).convert("RGB")
                fraction = tissue_fraction(
                    create_tissue_mask(tile, intensity_threshold=int(config["intensity_threshold"]))
                )
                if fraction >= float(config["min_tissue_fraction"]):
                    lesion_tissue_eligible += 1
                    lesion_eligible_fractions.append(float(fraction))
                    lesion_bins.add(
                        spatial_bin(x_zero, y_zero, slide_width, slide_height, grid_rows, grid_columns)
                    )
    finally:
        slide.close()

    selected = np.load(selected_path, allow_pickle=False)
    allocations = {
        (int(item["row"]), int(item["column"])): int(item["allocated_count"])
        for item in v2_row["bin_allocations"]
    }
    selected_bin_counts: dict[tuple[int, int], int] = {}
    selected_lesion_hits = 0
    for x_value, y_value in selected:
        key = spatial_bin(int(x_value), int(y_value), slide_width, slide_height, grid_rows, grid_columns)
        selected_bin_counts[key] = selected_bin_counts.get(key, 0) + 1
        if tile_hits_any_polygon(int(x_value), int(y_value), footprint, polygons):
            selected_lesion_hits += 1

    lesion_bin_allocated = sum(allocations.get(key, 0) for key in lesion_bins)
    lesion_bin_selected = sum(selected_bin_counts.get(key, 0) for key in lesion_bins)
    result = {
        "slide": row["slide"],
        "split": row["split"],
        "candidates_examined": candidates_examined,
        "annotation_polygon_count": len(polygons),
        "lesion_grid_candidate_count": lesion_grid_candidates,
        "lesion_tissue_eligible_candidate_count": lesion_tissue_eligible,
        "lesion_eligible_tissue_fraction_min": min(lesion_eligible_fractions) if lesion_eligible_fractions else None,
        "lesion_eligible_tissue_fraction_max": max(lesion_eligible_fractions) if lesion_eligible_fractions else None,
        "lesion_eligible_tissue_fraction_mean": float(np.mean(lesion_eligible_fractions)) if lesion_eligible_fractions else None,
        "lesion_candidate_bin_count": len(lesion_bins),
        "lesion_candidate_bins": [list(key) for key in sorted(lesion_bins)],
        "lesion_bin_allocated_tile_count": lesion_bin_allocated,
        "lesion_bin_selected_tile_count": lesion_bin_selected,
        "selected_lesion_intersecting_tile_count": selected_lesion_hits,
        "nearest_selected_center_to_annotation_point_distance": minimum_center_distance(selected, polygons, footprint),
        "selected_tile_count": int(len(selected)),
        "test_slide_loaded": False,
    }
    result["failure_category"] = categorize(result)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = []
    for row in rows:
        copy = dict(row)
        copy["lesion_candidate_bins"] = json.dumps(copy["lesion_candidate_bins"])
        flat.append(copy)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    processing = json.loads(project_path(config["processing_manifest"]).read_text(encoding="utf-8"))
    v2_manifest = json.loads(project_path(config["spatial_v2_manifest"]).read_text(encoding="utf-8"))
    processing_rows = {row["slide"]: row for row in processing["slides"]}
    v2_rows = {row["slide"]: row for row in v2_manifest["slides"]}
    expected = list(config["expected_uncovered_slides"])
    allowed = set(config["allowed_splits"])
    prohibited = set(config["prohibited_splits"])

    results = []
    for slide_name in expected:
        row = processing_rows[slide_name]
        if row["split"] not in allowed or row["split"] in prohibited:
            raise RuntimeError(f"Prohibited split encountered: {slide_name}")
        print(f"Diagnosing {slide_name} ({row['split']})")
        results.append(
            diagnose_slide(
                row,
                v2_rows[slide_name],
                project_path(config["spatial_v2_coordinate_root"]) / f"{slide_name}_coordinates.npy",
                project_path(config["annotation_root"]) / f"{slide_name}.xml",
                config,
            )
        )

    categories: dict[str, int] = {}
    for row in results:
        categories[row["failure_category"]] = categories.get(row["failure_category"], 0) + 1
    output = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "scientific_scope": config["scientific_scope"],
        "development_slides_analyzed": len(results),
        "test_slides_loaded": 0,
        "model_outputs_generated": False,
        "failure_category_counts": categories,
        "slides": results,
        "passed": True,
    }
    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "spatial_v2_failure_analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(output_root / "spatial_v2_failure_analysis.csv", results)
    print(json.dumps({key: value for key, value in output.items() if key != "slides"}, indent=2))
    print("PASS: Development-only Spatial v2 failure analysis completed.")


if __name__ == "__main__":
    main()
