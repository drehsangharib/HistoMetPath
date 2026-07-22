"""Development-only v2/v3 union upper-bound and consensus loss attribution.

The audit uses frozen coordinate sets and annotations after selection. It does
not create a new sampler, load test slides, generate embeddings, or produce
model outputs. For every development tumor slide, it measures the full parent
union and attributes lesion coordinates removed by the 300-tile consensus cap
to shared, v2-only, or v3-only coordinate lineage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from analysis.audit_camelyon16_lesion_coverage import (
    parse_polygons,
    rectangle_intersects_polygon,
)
from core.wsi.run_camelyon16_batch_pipeline import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v2/v3 union upper bound and consensus pruning losses."
    )
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_consensus_loss_attribution.yaml",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_set(path: Path) -> set[tuple[int, int]]:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 2:
        raise RuntimeError(f"Invalid coordinate array: {path}")
    return {tuple(map(int, point)) for point in array}


def lesion_coordinates(
    coordinates: set[tuple[int, int]],
    polygons: list[np.ndarray],
    footprint: float,
) -> set[tuple[int, int]]:
    output = set()
    for x_value, y_value in coordinates:
        if any(
            rectangle_intersects_polygon(
                x_value,
                y_value,
                x_value + footprint,
                y_value + footprint,
                polygon,
            )
            for polygon in polygons
        ):
            output.add((x_value, y_value))
    return output


def covered_polygon_fraction(
    coordinates: set[tuple[int, int]],
    polygons: list[np.ndarray],
    footprint: float,
) -> float:
    covered = 0
    for polygon in polygons:
        if any(
            rectangle_intersects_polygon(
                x_value,
                y_value,
                x_value + footprint,
                y_value + footprint,
                polygon,
            )
            for x_value, y_value in coordinates
        ):
            covered += 1
    return covered / len(polygons)


def audit_slide(
    processing_row: dict,
    consensus_row: dict,
    roots: dict[str, Path],
    annotation_root: Path,
    tile_size: int,
) -> dict:
    slide_name = processing_row["slide"]
    v2 = coordinate_set(roots["v2"] / f"{slide_name}_coordinates.npy")
    v3 = coordinate_set(roots["v3"] / f"{slide_name}_coordinates.npy")
    consensus = coordinate_set(
        roots["consensus"] / f"{slide_name}_coordinates.npy"
    )
    shared = v2 & v3
    v2_unique = v2 - v3
    v3_unique = v3 - v2
    union = v2 | v3

    if not consensus.issubset(union):
        raise RuntimeError(f"{slide_name}: consensus lacks parent lineage")

    polygons = parse_polygons(annotation_root / f"{slide_name}.xml")
    footprint = float(tile_size) * float(processing_row["selected_downsample"])

    v2_lesion = lesion_coordinates(v2, polygons, footprint)
    v3_lesion = lesion_coordinates(v3, polygons, footprint)
    union_lesion = lesion_coordinates(union, polygons, footprint)
    consensus_lesion = lesion_coordinates(consensus, polygons, footprint)
    discarded_lesion = union_lesion - consensus

    discarded_shared = discarded_lesion & shared
    discarded_v2_unique = discarded_lesion & v2_unique
    discarded_v3_unique = discarded_lesion & v3_unique

    if discarded_shared:
        raise RuntimeError(
            f"{slide_name}: shared coordinates should have been preserved"
        )

    if discarded_v2_unique and discarded_v3_unique:
        loss_source = "both_unique_sources_pruned"
    elif discarded_v2_unique:
        loss_source = "v2_unique_pruned"
    elif discarded_v3_unique:
        loss_source = "v3_unique_pruned"
    elif discarded_lesion:
        loss_source = "unexpected_shared_pruning"
    else:
        loss_source = "no_lesion_coordinate_loss"

    return {
        "slide": slide_name,
        "split": processing_row["split"],
        "annotation_polygon_count": len(polygons),
        "v2_coordinate_count": len(v2),
        "v3_coordinate_count": len(v3),
        "shared_coordinate_count": len(shared),
        "v2_unique_coordinate_count": len(v2_unique),
        "v3_unique_coordinate_count": len(v3_unique),
        "union_coordinate_count": len(union),
        "consensus_coordinate_count": len(consensus),
        "v2_lesion_tile_count": len(v2_lesion),
        "v3_lesion_tile_count": len(v3_lesion),
        "union_lesion_tile_count": len(union_lesion),
        "consensus_lesion_tile_count": len(consensus_lesion),
        "discarded_union_lesion_tile_count": len(discarded_lesion),
        "discarded_v2_unique_lesion_tile_count": len(discarded_v2_unique),
        "discarded_v3_unique_lesion_tile_count": len(discarded_v3_unique),
        "union_has_lesion": bool(union_lesion),
        "consensus_has_lesion": bool(consensus_lesion),
        "consensus_lost_parent_bag_coverage": bool(union_lesion and not consensus_lesion),
        "v2_polygon_fraction": covered_polygon_fraction(v2, polygons, footprint),
        "v3_polygon_fraction": covered_polygon_fraction(v3, polygons, footprint),
        "union_polygon_fraction": covered_polygon_fraction(union, polygons, footprint),
        "consensus_polygon_fraction": covered_polygon_fraction(
            consensus, polygons, footprint
        ),
        "shared_selected": int(consensus_row["shared_selected"]),
        "v2_unique_selected": int(consensus_row["v2_unique_selected"]),
        "v3_unique_selected": int(consensus_row["v3_unique_selected"]),
        "loss_source": loss_source,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))

    processing_path = project_path(config["processing_manifest"])
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    consensus_manifest_path = project_path(config["consensus_manifest"])
    consensus_manifest = json.loads(
        consensus_manifest_path.read_text(encoding="utf-8")
    )

    allowed = set(config["allowed_splits"])
    prohibited = set(config["prohibited_splits"])
    tumor_rows = [
        row
        for row in processing["slides"]
        if row["label"] == "tumor" and row["split"] in allowed
    ]
    if len(tumor_rows) != 18:
        raise RuntimeError(f"Expected 18 development tumor slides; found {len(tumor_rows)}")
    if any(row["split"] in prohibited for row in tumor_rows):
        raise RuntimeError("Prohibited test split entered loss attribution")

    consensus_rows = {
        row["slide"]: row for row in consensus_manifest["slides"]
    }
    roots = {
        "v2": project_path(config["v2_coordinate_root"]),
        "v3": project_path(config["v3_coordinate_root"]),
        "consensus": project_path(config["consensus_coordinate_root"]),
    }
    annotation_root = project_path(config["annotation_root"])

    results = [
        audit_slide(
            row,
            consensus_rows[row["slide"]],
            roots,
            annotation_root,
            int(config["tile_size"]),
        )
        for row in sorted(tumor_rows, key=lambda item: item["slide"])
    ]

    observed_losses = sorted(
        row["slide"]
        for row in results
        if row["consensus_lost_parent_bag_coverage"]
    )
    expected_losses = sorted(config["expected_consensus_parent_losses"])
    if observed_losses != expected_losses:
        raise RuntimeError(
            f"Observed consensus losses {observed_losses} differ from expected {expected_losses}"
        )

    summary = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "scientific_scope": config["scientific_scope"],
        "development_tumor_slides": 18,
        "test_slides_loaded": 0,
        "model_outputs_generated": False,
        "v2_bags_with_lesion": sum(row["v2_lesion_tile_count"] > 0 for row in results),
        "v3_bags_with_lesion": sum(row["v3_lesion_tile_count"] > 0 for row in results),
        "union_upper_bound_bags_with_lesion": sum(row["union_has_lesion"] for row in results),
        "consensus_bags_with_lesion": sum(row["consensus_has_lesion"] for row in results),
        "union_total_lesion_tiles": sum(row["union_lesion_tile_count"] for row in results),
        "consensus_total_lesion_tiles": sum(row["consensus_lesion_tile_count"] for row in results),
        "union_mean_polygon_coverage": float(np.mean([row["union_polygon_fraction"] for row in results])),
        "consensus_mean_polygon_coverage": float(np.mean([row["consensus_polygon_fraction"] for row in results])),
        "consensus_parent_coverage_losses": observed_losses,
        "discarded_v2_unique_lesion_tiles": sum(row["discarded_v2_unique_lesion_tile_count"] for row in results),
        "discarded_v3_unique_lesion_tiles": sum(row["discarded_v3_unique_lesion_tile_count"] for row in results),
        "final_test_lock_sha256": sha256_file(project_path(config["final_test_lock"])),
        "slides": results,
        "passed": True,
    }

    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "consensus_loss_attribution.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "consensus_loss_attribution.csv", results)

    concise = {key: value for key, value in summary.items() if key != "slides"}
    print(json.dumps(concise, indent=2))
    print("PASS: Development-only consensus loss-attribution audit completed.")


if __name__ == "__main__":
    main()
