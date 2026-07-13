"""
Split-aware PCAM patch-embedding extractor for HistoMetPath.

This utility:

- uses the repository's PCamHDF5Dataset interface;
- uses HistoMetPathModel and extracts features from model.backbone;
- loads a PyTorch Lightning checkpoint;
- keeps PCAM train, validation, and test splits separate;
- creates deterministic split-specific embedding and label arrays.

The resulting embeddings are patch-level features. They are not native WSI
representations.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import PCamHDF5Dataset
from models import HistoMetPathModel


SPLIT_TO_FILE_PREFIX = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}

DEFAULT_MAX_SAMPLES = {
    "train": 20_000,
    "valid": 5_000,
    "test": 5_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract split-specific PCAM patch embeddings "
            "using a trained HistoMetPath checkpoint."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(SPLIT_TO_FILE_PREFIX),
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a Lightning .ckpt file. If omitted, "
            "the newest checkpoint under logs/ is used."
        ),
    )

    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum patches to process. Use 0 for the full split. "
            "Defaults: train=20000, valid=5000, test=5000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--use-stain-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def find_checkpoint(
    requested_checkpoint: str | None,
) -> Path:
    if requested_checkpoint:
        checkpoint_path = Path(requested_checkpoint)

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                PROJECT_ROOT / checkpoint_path
            ).resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        return checkpoint_path

    checkpoint_candidates = sorted(
        (PROJECT_ROOT / "logs").rglob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoint_candidates:
        raise FileNotFoundError(
            "No Lightning checkpoint was found under "
            f"{PROJECT_ROOT / 'logs'}. Train the patch model "
            "first or supply --checkpoint."
        )

    return checkpoint_candidates[0]


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint,
        )
    else:
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint state_dict is not a dictionary."
        )

    cleaned: dict[str, torch.Tensor] = {}

    prefixes = (
        "model.",
        "module.",
    )

    for original_key, value in state_dict.items():
        key = original_key

        prefix_removed = True

        while prefix_removed:
            prefix_removed = False

            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    prefix_removed = True

        cleaned[key] = value

    return cleaned


def load_trained_model(
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

    state_dict = extract_state_dict(checkpoint)

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    critical_missing = [
        key
        for key in missing_keys
        if key.startswith("backbone.")
    ]

    if critical_missing:
        preview = ", ".join(critical_missing[:10])

        raise RuntimeError(
            "Checkpoint did not provide the required backbone "
            f"weights. Missing keys include: {preview}"
        )

    if unexpected_keys:
        print(
            "Checkpoint note: ignored unexpected keys: "
            f"{len(unexpected_keys)}"
        )

    if missing_keys:
        print(
            "Checkpoint note: noncritical missing keys: "
            f"{len(missing_keys)}"
        )

    model.to(device)
    model.eval()

    return model


def build_dataset(
    split: str,
    use_stain_norm: bool,
) -> PCamHDF5Dataset:
    prefix = SPLIT_TO_FILE_PREFIX[split]
    pcam_root = PROJECT_ROOT / "data" / "pcam"

    image_path = (
        pcam_root
        / f"camelyonpatch_level_2_split_{prefix}_x.h5"
    )

    label_path = (
        pcam_root
        / f"camelyonpatch_level_2_split_{prefix}_y.h5"
    )

    missing = [
        str(path)
        for path in (image_path, label_path)
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required PCAM files are missing:\n"
            f"{formatted}"
        )

    return PCamHDF5Dataset(
        str(image_path),
        str(label_path),
        build_transform(),
        use_stain_norm=use_stain_norm,
    )


def limit_dataset(
    dataset: PCamHDF5Dataset,
    split: str,
    max_samples: int | None,
) -> tuple[Subset | PCamHDF5Dataset, int]:
    requested = (
        DEFAULT_MAX_SAMPLES[split]
        if max_samples is None
        else max_samples
    )

    if requested < 0:
        raise ValueError(
            "--max-samples must be zero or positive."
        )

    if requested == 0:
        selected_count = len(dataset)
    else:
        selected_count = min(
            requested,
            len(dataset),
        )

    if selected_count < len(dataset):
        return (
            Subset(
                dataset,
                range(selected_count),
            ),
            selected_count,
        )

    return dataset, selected_count


def normalize_labels(
    labels: torch.Tensor | np.ndarray,
) -> np.ndarray:
    if torch.is_tensor(labels):
        array = labels.detach().cpu().numpy()
    else:
        array = np.asarray(labels)

    return array.reshape(-1).astype(
        np.int64,
        copy=False,
    )


def extract_embeddings(
    model: HistoMetPathModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            if not isinstance(batch, (tuple, list)):
                raise TypeError(
                    "Dataset batch must be a tuple or list."
                )

            if len(batch) < 2:
                raise ValueError(
                    "Dataset batch must contain images and labels."
                )

            images = batch[0].to(
                device,
                non_blocking=True,
            )

            labels = normalize_labels(
                batch[1]
            )

            features = model.backbone(images)

            if features.ndim > 2:
                features = torch.flatten(
                    features,
                    start_dim=1,
                )

            embedding_batches.append(
                features.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

            label_batches.append(labels)

            if (
                batch_index == 1
                or batch_index % 25 == 0
                or batch_index == len(loader)
            ):
                processed = sum(
                    len(batch_array)
                    for batch_array
                    in label_batches
                )

                print(
                    f"Processed {processed} patches "
                    f"({batch_index}/{len(loader)} batches)"
                )

    if not embedding_batches:
        raise RuntimeError(
            "Embedding extraction produced no batches."
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    labels = np.concatenate(
        label_batches,
        axis=0,
    )

    if len(embeddings) != len(labels):
        raise RuntimeError(
            "Embedding and label counts do not match."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embeddings contain NaN or infinite values."
        )

    return embeddings, labels


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = find_checkpoint(
        args.checkpoint
    )

    dataset = build_dataset(
        split=args.split,
        use_stain_norm=args.use_stain_norm,
    )

    selected_dataset, selected_count = limit_dataset(
        dataset=dataset,
        split=args.split,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = load_trained_model(
        checkpoint_path=checkpoint_path,
        backbone_name=args.backbone,
        device=device,
    )

    print("=" * 70)
    print("HistoMetPath split-aware embedding extraction")
    print(f"Split:          {args.split}")
    print(f"Device:         {device}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Backbone:       {args.backbone}")
    print(f"Stain norm:     {args.use_stain_norm}")
    print(f"Selected count: {selected_count}")
    print("=" * 70)

    embeddings, labels = extract_embeddings(
        model=model,
        loader=loader,
        device=device,
    )

    embedding_dir = PROJECT_ROOT / "embeddings"

    embedding_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        embedding_dir
        / f"{args.split}_embeddings.npy"
    )

    labels_path = (
        embedding_dir
        / f"{args.split}_labels.npy"
    )

    np.save(
        embeddings_path,
        embeddings,
        allow_pickle=False,
    )

    np.save(
        labels_path,
        labels,
        allow_pickle=False,
    )

    print()
    print("PASS: Split embedding extraction completed.")
    print(
        f"Embeddings: {embeddings_path} "
        f"shape={embeddings.shape}"
    )
    print(
        f"Labels:     {labels_path} "
        f"shape={labels.shape}"
    )
    print(
        f"Positive fraction: {labels.mean():.6f}"
    )


if __name__ == "__main__":
    main()
