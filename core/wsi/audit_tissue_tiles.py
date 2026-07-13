"""Audit CAMELYON16 tissue-tile outputs and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILE_ROOT = PROJECT_ROOT / "outputs" / "camelyon16" / "tissue_tiles"
MANIFEST_ROOT = PROJECT_ROOT / "outputs" / "camelyon16" / "tissue_manifests"
AUDIT_PATH = PROJECT_ROOT / "outputs" / "camelyon16" / "tissue_tile_audit.json"


def audit_slide(slide_name: str) -> dict:
    tile_directory = TILE_ROOT / slide_name
    manifest_path = MANIFEST_ROOT / f"{slide_name}.json"

    if not tile_directory.is_dir():
        raise FileNotFoundError(tile_directory)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    png_files = sorted(tile_directory.glob("*.png"))
    manifest_names = {row["tile_name"] for row in manifest}
    disk_names = {path.name for path in png_files}

    missing_on_disk = sorted(manifest_names - disk_names)
    unlisted_on_disk = sorted(disk_names - manifest_names)
    dimensions = set()
    channel_counts = set()
    mean_intensities = []

    for tile_path in png_files:
        with Image.open(tile_path) as image:
            rgb = image.convert("RGB")
            dimensions.add(rgb.size)
            channel_counts.add(len(rgb.getbands()))
            array = np.asarray(rgb, dtype=np.float32)
            mean_intensities.append(float(array.mean()))

    tissue_fractions = [
        float(row["tissue_fraction"])
        for row in manifest
        if "tissue_fraction" in row
    ]

    if missing_on_disk:
        raise RuntimeError(
            f"{slide_name}: manifest tiles missing on disk: {missing_on_disk[:5]}"
        )
    if unlisted_on_disk:
        raise RuntimeError(
            f"{slide_name}: unlisted tiles found: {unlisted_on_disk[:5]}"
        )
    if len(manifest) != len(png_files):
        raise RuntimeError(f"{slide_name}: manifest count and PNG count differ.")
    if dimensions != {(256, 256)}:
        raise RuntimeError(
            f"{slide_name}: unexpected tile dimensions: {sorted(dimensions)}"
        )
    if channel_counts != {3}:
        raise RuntimeError(
            f"{slide_name}: unexpected channel counts: {sorted(channel_counts)}"
        )

    return {
        "slide": slide_name,
        "tile_count": len(png_files),
        "manifest_count": len(manifest),
        "tile_dimensions": [256, 256],
        "channels": 3,
        "tissue_fraction_min": min(tissue_fractions) if tissue_fractions else None,
        "tissue_fraction_mean": (
            float(np.mean(tissue_fractions)) if tissue_fractions else None
        ),
        "tissue_fraction_max": max(tissue_fractions) if tissue_fractions else None,
        "mean_pixel_intensity": (
            float(np.mean(mean_intensities)) if mean_intensities else None
        ),
        "missing_on_disk": missing_on_disk,
        "unlisted_on_disk": unlisted_on_disk,
        "passed": True,
    }


def main() -> None:
    slide_names = ["normal_100", "tumor_100"]
    slide_audits = [audit_slide(slide_name) for slide_name in slide_names]

    output = {
        "schema_version": "1.0",
        "dataset": "CAMELYON16",
        "scientific_scope": (
            "two-slide real-WSI pipeline pilot; not a model-performance benchmark"
        ),
        "slide_count": len(slide_audits),
        "total_tile_count": sum(row["tile_count"] for row in slide_audits),
        "slides": slide_audits,
        "passed": all(row["passed"] for row in slide_audits),
    }

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print()
    print(f"Audit written to: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
