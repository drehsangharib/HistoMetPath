"""Annotation-independent spatially distributed CAMELYON16 sampler.

This development command scans tissue candidates across the full selected WSI
level, partitions the complete level-zero slide extent into a fixed spatial
grid, and retains the highest-tissue-fraction candidates per occupied bin.
Annotations are never loaded by this sampler. Test slides are explicitly
refused.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from pathlib import Path

import numpy as np
import openslide
import yaml

from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.tissue_mask import create_tissue_mask, tissue_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run development-only spatial WSI sampling.")
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_spatial_sampler_development.yaml",
    )
    parser.add_argument("--slides", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    required = {
        "processing_manifest", "output_root", "tile_size", "stride",
        "intensity_threshold", "min_tissue_fraction", "spatial_grid_rows",
        "spatial_grid_columns", "max_tiles_per_bin", "max_tiles_per_slide",
        "allowed_splits", "prohibited_splits",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    if "test" not in config["prohibited_splits"]:
        raise RuntimeError("The test split must be explicitly prohibited.")
    return config


def spatial_bin(
    x_zero: int,
    y_zero: int,
    slide_width: int,
    slide_height: int,
    grid_rows: int,
    grid_columns: int,
) -> tuple[int, int]:
    column = min(grid_columns - 1, int(x_zero * grid_columns / slide_width))
    row = min(grid_rows - 1, int(y_zero * grid_rows / slide_height))
    return row, column


def select_from_bins(
    candidates_by_bin: dict[tuple[int, int], list[tuple]],
    max_tiles_per_slide: int,
) -> list[dict]:
    selected = []
    for (row, column), heap in sorted(candidates_by_bin.items()):
        for item in sorted(heap, key=lambda value: (-value[0], value[2], value[1])):
            fraction, x_zero, y_zero = item
            selected.append(
                {
                    "x": int(x_zero),
                    "y": int(y_zero),
                    "tissue_fraction": float(fraction),
                    "spatial_bin_row": int(row),
                    "spatial_bin_column": int(column),
                }
            )
    if len(selected) > max_tiles_per_slide:
        selected = selected[:max_tiles_per_slide]
    selected.sort(key=lambda row: (row["spatial_bin_row"], row["spatial_bin_column"], -row["tissue_fraction"], row["y"], row["x"]))
    return selected


def sample_slide(row: dict, config: dict) -> dict:
    slide_path = Path(row["path"])
    slide = openslide.OpenSlide(str(slide_path))
    started = time.time()
    try:
        level = int(row["selected_level"])
        downsample = float(row["selected_downsample"])
        level_width, level_height = slide.level_dimensions[level]
        slide_width, slide_height = slide.dimensions
        tile_size = int(config["tile_size"])
        stride = int(config["stride"])
        grid_rows = int(config["spatial_grid_rows"])
        grid_columns = int(config["spatial_grid_columns"])
        max_per_bin = int(config["max_tiles_per_bin"])
        threshold = float(config["min_tissue_fraction"])
        intensity = int(config["intensity_threshold"])
        candidates_by_bin: dict[tuple[int, int], list[tuple]] = {}
        candidates_examined = 0
        tissue_candidates = 0

        for y_level in range(0, level_height - tile_size + 1, stride):
            for x_level in range(0, level_width - tile_size + 1, stride):
                candidates_examined += 1
                x_zero = int(round(x_level * downsample))
                y_zero = int(round(y_level * downsample))
                tile = slide.read_region((x_zero, y_zero), level, (tile_size, tile_size)).convert("RGB")
                fraction = tissue_fraction(
                    create_tissue_mask(tile, intensity_threshold=intensity)
                )
                if fraction < threshold:
                    continue
                tissue_candidates += 1
                key = spatial_bin(
                    x_zero, y_zero, slide_width, slide_height, grid_rows, grid_columns
                )
                heap = candidates_by_bin.setdefault(key, [])
                item = (float(fraction), int(x_zero), int(y_zero))
                if len(heap) < max_per_bin:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    finally:
        slide.close()

    selected = select_from_bins(candidates_by_bin, int(config["max_tiles_per_slide"]))
    if not selected:
        raise RuntimeError(f"No tissue tiles selected for {row['slide']}")
    coordinates = np.asarray([[item["x"], item["y"]] for item in selected], dtype=np.int64)
    fractions = np.asarray([item["tissue_fraction"] for item in selected], dtype=np.float32)
    return {
        "slide": row["slide"],
        "label": row["label"],
        "split": row["split"],
        "path": row["path"],
        "selected_level": level,
        "selected_downsample": downsample,
        "effective_mpp": row["effective_mpp"],
        "candidates_examined": candidates_examined,
        "tissue_candidate_count": tissue_candidates,
        "occupied_bin_count": len(candidates_by_bin),
        "grid_bin_count": grid_rows * grid_columns,
        "selected_tile_count": len(selected),
        "mean_tissue_fraction": float(fractions.mean()),
        "coordinate_x_min": int(coordinates[:, 0].min()),
        "coordinate_x_max": int(coordinates[:, 0].max()),
        "coordinate_y_min": int(coordinates[:, 1].min()),
        "coordinate_y_max": int(coordinates[:, 1].max()),
        "elapsed_seconds": float(time.time() - started),
        "tiles": selected,
        "status": "complete",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    flattened = [{key: value for key, value in row.items() if key != "tiles"} for row in rows]
    fieldnames = list(flattened[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    manifest_path = project_path(config["processing_manifest"])
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = set(config["allowed_splits"])
    prohibited = set(config["prohibited_splits"])
    if allowed & prohibited:
        raise RuntimeError("Allowed and prohibited split definitions overlap.")

    rows = [row for row in source["slides"] if row["split"] in allowed]
    if any(row["split"] in prohibited for row in rows):
        raise RuntimeError("Prohibited split entered sampler development.")
    if len(rows) != 36:
        raise RuntimeError(f"Expected 36 development slides; found {len(rows)}")
    if args.slides:
        requested = set(args.slides)
        rows = [row for row in rows if row["slide"] in requested]
        missing = sorted(requested - {row["slide"] for row in rows})
        if missing:
            raise RuntimeError(f"Requested slides unavailable or prohibited: {missing}")

    output_root = project_path(config["output_root"])
    coordinate_root = output_root / "coordinates"
    output_root.mkdir(parents=True, exist_ok=True)
    coordinate_root.mkdir(parents=True, exist_ok=True)
    manifest_output = output_root / "spatial_sampling_manifest.json"
    prior = {}
    if manifest_output.is_file() and not args.force:
        prior_data = json.loads(manifest_output.read_text(encoding="utf-8"))
        prior = {row["slide"]: row for row in prior_data.get("slides", [])}

    results = []
    for index, row in enumerate(rows, start=1):
        if row["slide"] in prior and prior[row["slide"]].get("status") == "complete":
            print(f"Skipping completed slide {row['slide']}")
            results.append(prior[row["slide"]])
            continue
        print(f"Spatial sampling {index}/{len(rows)}: {row['slide']} ({row['split']})")
        result = sample_slide(row, config)
        np.save(
            coordinate_root / f"{row['slide']}_coordinates.npy",
            np.asarray([[item["x"], item["y"]] for item in result["tiles"]], dtype=np.int64),
            allow_pickle=False,
        )
        np.save(
            coordinate_root / f"{row['slide']}_tissue_fractions.npy",
            np.asarray([item["tissue_fraction"] for item in result["tiles"]], dtype=np.float32),
            allow_pickle=False,
        )
        results.append(result)
        payload = {
            "schema_version": "1.0",
            "dataset": config["dataset"],
            "scientific_scope": config["scientific_scope"],
            "development_slide_count": len(rows),
            "test_slides_loaded": 0,
            "prohibited_splits": sorted(prohibited),
            "slides": results,
            "completed_count": len(results),
            "passed": len(results) == len(rows),
        }
        manifest_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(output_root / "spatial_sampling_summary.csv", results)

    print(json.dumps({
        "development_slide_count": len(rows),
        "completed_count": len(results),
        "test_slides_loaded": 0,
        "passed": len(results) == len(rows),
    }, indent=2))
    print("PASS: Annotation-independent development-only spatial sampling completed.")


if __name__ == "__main__":
    main()
