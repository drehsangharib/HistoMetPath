"""Extract real CAMELYON16 tissue-tile embeddings into slide-specific bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import HistoMetPathModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLIDES = ("normal_100", "tumor_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from audited CAMELYON16 tissue tiles."
    )
    parser.add_argument("--slides", nargs="+", default=list(DEFAULT_SLIDES))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--audit-path",
        default="outputs/camelyon16/tissue_tile_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        default="embeddings/camelyon16",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_checkpoint(requested: str | None) -> Path:
    if requested:
        path = resolve_project_path(requested)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = sorted(
        (PROJECT_ROOT / "logs").rglob("*.ckpt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found under logs/. Supply --checkpoint explicitly."
        )
    return candidates[0]


def clean_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    raw = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint state_dict must be a dictionary.")

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


def load_model(
    checkpoint_path: Path,
    backbone_name: str,
    device: torch.device,
) -> HistoMetPathModel:
    model = HistoMetPathModel(
        backbone=backbone_name,
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
    incompatible = model.load_state_dict(
        clean_state_dict(checkpoint),
        strict=False,
    )
    critical_missing = [
        key for key in incompatible.missing_keys if key.startswith("backbone.")
    ]
    if critical_missing:
        preview = ", ".join(critical_missing[:10])
        raise RuntimeError(
            "Checkpoint is missing required backbone weights: " + preview
        )
    if incompatible.unexpected_keys:
        print(
            "Checkpoint note: ignored unexpected keys: "
            f"{len(incompatible.unexpected_keys)}"
        )
    if incompatible.missing_keys:
        print(
            "Checkpoint note: noncritical missing keys: "
            f"{len(incompatible.missing_keys)}"
        )
    model.to(device)
    model.eval()
    return model


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ManifestTileDataset(Dataset):
    def __init__(
        self,
        tile_directory: Path,
        manifest_path: Path,
        transform: transforms.Compose,
    ) -> None:
        self.tile_directory = tile_directory
        self.rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.transform = transform
        if not isinstance(self.rows, list) or not self.rows:
            raise RuntimeError(f"Empty or invalid tile manifest: {manifest_path}")
        for row in self.rows:
            if "tile_name" not in row or "x" not in row or "y" not in row:
                raise KeyError("Each manifest row requires tile_name, x, and y.")
            tile_path = self.tile_directory / row["tile_name"]
            if not tile_path.is_file():
                raise FileNotFoundError(tile_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        tile_path = self.tile_directory / row["tile_name"]
        with Image.open(tile_path) as image:
            tensor = self.transform(image.convert("RGB"))
        coordinates = torch.tensor([int(row["x"]), int(row["y"])])
        return tensor, coordinates, row["tile_name"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(audit_path: Path, requested_slides: list[str]) -> dict:
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("Tissue-tile audit did not pass.")
    rows = {row["slide"]: row for row in audit.get("slides", [])}
    missing = [slide for slide in requested_slides if slide not in rows]
    if missing:
        raise RuntimeError(f"Slides missing from audit: {missing}")
    for slide_name in requested_slides:
        if rows[slide_name].get("passed") is not True:
            raise RuntimeError(f"Slide audit failed: {slide_name}")
    return audit


def infer_feature_dimension(model: HistoMetPathModel, image_size: int, device: torch.device) -> int:
    with torch.inference_mode():
        sample = torch.zeros(1, 3, image_size, image_size, device=device)
        features = model.backbone(sample)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
    return int(features.shape[1])


def extract_one_slide(
    slide_name: str,
    model: HistoMetPathModel,
    transform: transforms.Compose,
    output_directory: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    tile_directory = (
        PROJECT_ROOT / "outputs" / "camelyon16" / "tissue_tiles" / slide_name
    )
    manifest_path = (
        PROJECT_ROOT
        / "outputs"
        / "camelyon16"
        / "tissue_manifests"
        / f"{slide_name}.json"
    )
    dataset = ManifestTileDataset(tile_directory, manifest_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    embedding_batches: list[np.ndarray] = []
    coordinate_batches: list[np.ndarray] = []
    tile_names: list[str] = []

    with torch.inference_mode():
        for batch_number, (images, coordinates, names) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = model.backbone(images)
            if features.ndim > 2:
                features = torch.flatten(features, start_dim=1)
            embedding_batches.append(
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            coordinate_batches.append(coordinates.numpy().astype(np.int64, copy=False))
            tile_names.extend(list(names))
            processed = sum(len(batch) for batch in embedding_batches)
            print(
                f"{slide_name}: processed {processed}/{len(dataset)} tiles "
                f"({batch_number}/{len(loader)} batches)"
            )

    embeddings = np.concatenate(embedding_batches, axis=0)
    coordinates = np.concatenate(coordinate_batches, axis=0)
    if len(embeddings) != len(dataset) or len(coordinates) != len(dataset):
        raise RuntimeError(f"{slide_name}: output row count mismatch.")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"{slide_name}: embeddings contain nonfinite values.")

    output_directory.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_directory / f"{slide_name}_embeddings.npy"
    coordinates_path = output_directory / f"{slide_name}_coordinates.npy"
    names_path = output_directory / f"{slide_name}_tile_names.json"
    np.save(embeddings_path, embeddings, allow_pickle=False)
    np.save(coordinates_path, coordinates, allow_pickle=False)
    names_path.write_text(json.dumps(tile_names, indent=2), encoding="utf-8")

    return {
        "slide": slide_name,
        "tile_count": int(len(dataset)),
        "embedding_shape": list(embeddings.shape),
        "coordinate_shape": list(coordinates.shape),
        "embeddings_path": str(embeddings_path),
        "coordinates_path": str(coordinates_path),
        "tile_names_path": str(names_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "coordinates_sha256": sha256_file(coordinates_path),
        "tile_names_sha256": sha256_file(names_path),
        "embedding_mean": float(embeddings.mean()),
        "embedding_standard_deviation": float(embeddings.std()),
        "passed": True,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    audit_path = resolve_project_path(args.audit_path)
    output_directory = resolve_project_path(args.output_dir)
    validate_audit(audit_path, args.slides)
    checkpoint_path = find_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, args.backbone, device)
    transform = build_transform(args.image_size)
    feature_dimension = infer_feature_dimension(model, args.image_size, device)

    print("=" * 72)
    print("HistoMetPath real CAMELYON16 tissue-tile embedding extraction")
    print(f"Slides:            {', '.join(args.slides)}")
    print(f"Device:            {device}")
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Backbone:          {args.backbone}")
    print(f"Input image size:  {args.image_size}")
    print(f"Feature dimension: {feature_dimension}")
    print("=" * 72)

    slide_results = [
        extract_one_slide(
            slide_name=slide_name,
            model=model,
            transform=transform,
            output_directory=output_directory,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for slide_name in args.slides
    ]

    manifest = {
        "schema_version": "1.0",
        "dataset": "CAMELYON16",
        "scientific_scope": (
            "two-slide real-WSI tile-embedding pipeline pilot; "
            "not a model-performance benchmark"
        ),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "backbone": args.backbone,
        "image_size": int(args.image_size),
        "feature_dimension": feature_dimension,
        "device": str(device),
        "total_tile_count": sum(row["tile_count"] for row in slide_results),
        "slides": slide_results,
        "passed": all(row["passed"] for row in slide_results),
    }
    manifest_path = (
        PROJECT_ROOT / "outputs" / "camelyon16" / "embedding_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print(json.dumps(manifest, indent=2))
    print()
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
