"""Configuration-driven CAMELYON16 batch preprocessing and embedding pipeline.

The pipeline discovers local training WSIs, builds a deterministic stratified
slide split, selects a pyramid level by physical resolution, samples tissue-rich
tiles, extracts trained CNN embeddings on the fly, and creates a slide-bag
registry. Runtime data remain outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import torch
import yaml
from PIL import Image
from torchvision import transforms

from core.wsi.tissue_mask import create_tissue_mask, tissue_fraction
from models import HistoMetPathModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAMELYON16 batch pilot.")
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_batch_pilot.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=["registry", "process", "all"],
        default="all",
    )
    parser.add_argument("--slides", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "data_root",
        "output_root",
        "embedding_root",
        "checkpoint",
        "seed",
        "target_mpp",
        "tile_size",
        "stride",
        "min_tissue_fraction",
        "max_tiles_per_slide",
        "split_counts_per_class",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Config is missing required keys: {missing}")
    return config


def classify_slide(path: Path) -> str:
    if path.stem.startswith("normal_"):
        return "normal"
    if path.stem.startswith("tumor_"):
        return "tumor"
    raise ValueError(f"Cannot infer label from slide name: {path.name}")


def discover_slides(data_root: Path) -> list[dict]:
    slides = []
    for path in sorted(data_root.rglob("*.tif")):
        if path.is_file():
            slides.append(
                {
                    "slide": path.stem,
                    "label": classify_slide(path),
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not slides:
        raise RuntimeError(f"No TIFF slides found under {data_root}")
    names = [row["slide"] for row in slides]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate slide identifiers were discovered.")
    return slides


def stratified_split(
    slides: list[dict],
    split_counts: dict,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    assigned: list[dict] = []
    required_per_class = sum(int(value) for value in split_counts.values())

    for label in ("normal", "tumor"):
        class_rows = [dict(row) for row in slides if row["label"] == label]
        if len(class_rows) != required_per_class:
            raise RuntimeError(
                f"Expected {required_per_class} {label} slides, found {len(class_rows)}."
            )
        order = rng.permutation(len(class_rows))
        shuffled = [class_rows[index] for index in order]
        cursor = 0
        for split_name in ("train", "validation", "test"):
            count = int(split_counts[split_name])
            for row in shuffled[cursor : cursor + count]:
                row["split"] = split_name
                assigned.append(row)
            cursor += count

    assigned.sort(key=lambda row: row["slide"])
    return assigned


def read_wsi_metadata(path: Path) -> dict:
    slide = openslide.OpenSlide(str(path))
    try:
        properties = slide.properties
        mpp_x = properties.get("openslide.mpp-x")
        mpp_y = properties.get("openslide.mpp-y")
        if mpp_x is None or mpp_y is None:
            raise RuntimeError(f"Missing microns-per-pixel metadata: {path.name}")
        return {
            "width": int(slide.dimensions[0]),
            "height": int(slide.dimensions[1]),
            "level_count": int(slide.level_count),
            "level_dimensions": [list(item) for item in slide.level_dimensions],
            "level_downsamples": [float(item) for item in slide.level_downsamples],
            "mpp_x": float(mpp_x),
            "mpp_y": float(mpp_y),
            "vendor": properties.get("openslide.vendor"),
            "quickhash": properties.get("openslide.quickhash-1"),
        }
    finally:
        slide.close()


def select_level(
    mpp_x: float,
    mpp_y: float,
    downsamples: list[float],
    target_mpp: float,
) -> tuple[int, float, float]:
    base_mpp = (mpp_x + mpp_y) / 2.0
    effective = [base_mpp * value for value in downsamples]
    level = min(
        range(len(effective)),
        key=lambda index: abs(math.log(effective[index] / target_mpp)),
    )
    return level, float(downsamples[level]), float(effective[level])


def build_transform(tile_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def clean_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    raw = checkpoint.get("state_dict", checkpoint)
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in raw.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in ("model.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_model(config: dict, device: torch.device) -> tuple[HistoMetPathModel, Path]:
    checkpoint_path = project_path(config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = HistoMetPathModel(
        backbone=config.get("backbone", "resnet18"),
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=512,
        num_classes=1,
        dropout=0.3,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    incompatible = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    critical = [key for key in incompatible.missing_keys if key.startswith("backbone.")]
    if critical:
        raise RuntimeError(f"Checkpoint is missing backbone keys: {critical[:5]}")
    model.to(device)
    model.eval()
    return model, checkpoint_path


def batched_features(
    model: HistoMetPathModel,
    tensors: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        features = model.backbone(batch)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
    return features.detach().cpu().numpy().astype(np.float32, copy=False)


def process_slide(
    row: dict,
    config: dict,
    model: HistoMetPathModel,
    device: torch.device,
    embedding_root: Path,
) -> dict:
    path = Path(row["path"])
    metadata = read_wsi_metadata(path)
    level, downsample, effective_mpp = select_level(
        metadata["mpp_x"],
        metadata["mpp_y"],
        metadata["level_downsamples"],
        float(config["target_mpp"]),
    )
    tile_size = int(config["tile_size"])
    stride = int(config["stride"])
    max_tiles = int(config["max_tiles_per_slide"])
    min_fraction = float(config["min_tissue_fraction"])
    intensity_threshold = int(config.get("intensity_threshold", 220))
    batch_size = int(config.get("batch_size", 32))
    level_width, level_height = metadata["level_dimensions"][level]
    transform = build_transform(tile_size)

    tensors: list[torch.Tensor] = []
    pending_coordinates: list[list[int]] = []
    coordinates: list[list[int]] = []
    fractions: list[float] = []
    feature_batches: list[np.ndarray] = []
    candidates_examined = 0

    slide = openslide.OpenSlide(str(path))
    try:
        stop = False
        for y_level in range(0, level_height - tile_size + 1, stride):
            for x_level in range(0, level_width - tile_size + 1, stride):
                candidates_examined += 1
                x_zero = int(round(x_level * downsample))
                y_zero = int(round(y_level * downsample))
                tile = slide.read_region(
                    (x_zero, y_zero),
                    level,
                    (tile_size, tile_size),
                ).convert("RGB")
                fraction = tissue_fraction(
                    create_tissue_mask(tile, intensity_threshold=intensity_threshold)
                )
                if fraction < min_fraction:
                    continue
                tensors.append(transform(tile))
                pending_coordinates.append([x_zero, y_zero])
                fractions.append(float(fraction))

                if len(tensors) >= batch_size:
                    feature_batches.append(batched_features(model, tensors, device))
                    coordinates.extend(pending_coordinates)
                    tensors = []
                    pending_coordinates = []

                accepted_count = len(coordinates) + len(pending_coordinates)
                if accepted_count >= max_tiles:
                    stop = True
                    break
            if stop:
                break

        if tensors:
            feature_batches.append(batched_features(model, tensors, device))
            coordinates.extend(pending_coordinates)
    finally:
        slide.close()

    if not feature_batches:
        raise RuntimeError(f"No tissue tiles accepted for {row['slide']}")

    embeddings = np.concatenate(feature_batches, axis=0)[:max_tiles]
    coordinate_array = np.asarray(coordinates[: len(embeddings)], dtype=np.int64)
    fraction_array = np.asarray(fractions[: len(embeddings)], dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(f"Unexpected embedding shape for {row['slide']}: {embeddings.shape}")
    if coordinate_array.shape != (len(embeddings), 2):
        raise RuntimeError(f"Coordinate mismatch for {row['slide']}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"Nonfinite embeddings for {row['slide']}")

    embedding_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = embedding_root / f"{row['slide']}_embeddings.npy"
    coordinates_path = embedding_root / f"{row['slide']}_coordinates.npy"
    fractions_path = embedding_root / f"{row['slide']}_tissue_fractions.npy"
    np.save(embeddings_path, embeddings, allow_pickle=False)
    np.save(coordinates_path, coordinate_array, allow_pickle=False)
    np.save(fractions_path, fraction_array, allow_pickle=False)

    return {
        **row,
        **metadata,
        "selected_level": level,
        "selected_downsample": downsample,
        "effective_mpp": effective_mpp,
        "target_mpp": float(config["target_mpp"]),
        "tile_size": tile_size,
        "stride": stride,
        "candidates_examined": candidates_examined,
        "tile_count": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "mean_tissue_fraction": float(fraction_array.mean()),
        "embeddings_path": str(embeddings_path),
        "coordinates_path": str(coordinates_path),
        "tissue_fractions_path": str(fractions_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "coordinates_sha256": sha256_file(coordinates_path),
        "tissue_fractions_sha256": sha256_file(fractions_path),
        "status": "complete",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_registry(config: dict, output_root: Path) -> list[dict]:
    slides = discover_slides(project_path(config["data_root"]))
    registry = stratified_split(
        slides,
        config["split_counts_per_class"],
        int(config["seed"]),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slide_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "slide_registry.csv", registry)
    split_summary = {
        split_name: {
            label: sum(
                row["split"] == split_name and row["label"] == label
                for row in registry
            )
            for label in ("normal", "tumor")
        }
        for split_name in ("train", "validation", "test")
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    return registry


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    output_root = project_path(config["output_root"])
    embedding_root = project_path(config["embedding_root"])
    registry = build_registry(config, output_root)

    if args.stage == "registry":
        print(json.dumps({"slides": len(registry), "stage": "registry", "passed": True}, indent=2))
        return

    selected = registry
    if args.slides:
        requested = set(args.slides)
        selected = [row for row in registry if row["slide"] in requested]
        missing = sorted(requested - {row["slide"] for row in selected})
        if missing:
            raise RuntimeError(f"Requested slides not in registry: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_path = load_model(config, device)
    progress_path = output_root / "processing_manifest.json"
    prior_rows: dict[str, dict] = {}
    if progress_path.is_file() and not args.force:
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        prior_rows = {row["slide"]: row for row in prior.get("slides", [])}

    results: list[dict] = []
    for index, row in enumerate(selected, start=1):
        if row["slide"] in prior_rows and prior_rows[row["slide"]].get("status") == "complete":
            print(f"Skipping completed slide {row['slide']}")
            results.append(prior_rows[row["slide"]])
            continue
        print(f"Processing slide {index}/{len(selected)}: {row['slide']}")
        result = process_slide(row, config, model, device, embedding_root)
        results.append(result)
        manifest = {
            "schema_version": "1.0",
            "dataset": config.get("dataset", "CAMELYON16"),
            "scientific_scope": config.get("scientific_scope"),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "device": str(device),
            "slides": results,
            "completed_count": len(results),
            "requested_count": len(selected),
            "passed": len(results) == len(selected),
        }
        progress_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_csv(output_root / "processing_manifest.csv", results)

    print()
    print(json.dumps({
        "registry_count": len(registry),
        "processed_count": len(results),
        "output_root": str(output_root),
        "embedding_root": str(embedding_root),
        "passed": len(results) == len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
