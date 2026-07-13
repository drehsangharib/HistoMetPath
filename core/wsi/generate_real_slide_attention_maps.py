"""Generate exploratory attention maps for real CAMELYON16 slide bags.

The Attention MIL model is trained on the existing controlled synthetic PCAM
bags for one deterministic seed, then applied to real WSI-derived bags. This
validates variable-size bag handling, coordinate alignment, top-tile retrieval,
and overlay generation. It is not a real-WSI performance benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from PIL import ImageDraw
from torch.optim import Adam

from analysis.attention_mil_v2 import AttentionMIL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLIDES = ("normal_100", "tumor_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real slide-bag attention-map mechanics."
    )
    parser.add_argument("--slides", nargs="+", default=list(DEFAULT_SLIDES))
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thumbnail-max-size", type=int, default=1600)
    parser.add_argument(
        "--output-dir",
        default="outputs/camelyon16/real_slide_attention",
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


def as_float32_bag(raw_bag) -> np.ndarray:
    array = np.asarray(raw_bag)
    if array.dtype == object:
        array = np.array(array.tolist(), dtype=np.float32)
    else:
        array = array.astype(np.float32, copy=False)
    array = np.ascontiguousarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D bag, found shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Bag contains nonfinite values.")
    return array


def load_controlled_training_bags(seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    directory = (
        PROJECT_ROOT
        / "embeddings"
        / "pseudo_slides_controlled"
        / "train"
        / f"seed_{seed}"
    )
    bags_path = directory / "bags.npy"
    labels_path = directory / "labels.npy"
    if not bags_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            "Controlled training bags are required. Missing: "
            f"{bags_path} or {labels_path}"
        )
    raw_bags = np.load(bags_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=False).reshape(-1).astype(np.float32)
    bags = [as_float32_bag(bag) for bag in raw_bags]
    if len(bags) != len(labels):
        raise RuntimeError("Controlled bag and label counts do not match.")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise RuntimeError("Controlled training labels must contain both classes.")
    return bags, labels


def train_attention_model(
    bags: list[np.ndarray],
    labels: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[AttentionMIL, list[float]]:
    in_dim = int(bags[0].shape[1])
    model = AttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(bags))
        total_loss = 0.0
        for index in order:
            tensor = torch.from_numpy(bags[index]).to(device)
            label = torch.tensor(labels[index], dtype=torch.float32, device=device)
            logit, _ = model(tensor)
            loss = F.binary_cross_entropy_with_logits(logit.reshape(1), label.reshape(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        mean_loss = total_loss / len(bags)
        epoch_losses.append(mean_loss)
        print(f"Training epoch {epoch + 1}/{epochs}: loss={mean_loss:.6f}")
    return model, epoch_losses


def score_slide_bag(
    model: AttentionMIL,
    embeddings: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    bag = as_float32_bag(embeddings)
    model.eval()
    with torch.inference_mode():
        logit, attention = model(torch.from_numpy(bag).to(device))
        probability = float(torch.sigmoid(logit).detach().cpu())
        weights = attention.detach().cpu().numpy().astype(np.float64, copy=False)
    if weights.shape != (len(bag),):
        raise RuntimeError(f"Attention shape mismatch: {weights.shape}")
    if not np.isfinite(weights).all():
        raise RuntimeError("Attention weights contain nonfinite values.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise RuntimeError(f"Attention weights sum to {weights.sum()}, not 1.")
    return probability, weights


def attention_entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log(positive)))


def find_slide_path(slide_name: str) -> Path:
    matches = sorted(
        path
        for path in (PROJECT_ROOT / "data" / "camelyon16").rglob("*")
        if path.is_file()
        and path.stem == slide_name
        and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one WSI for {slide_name}; found {len(matches)}."
        )
    return matches[0]


def color_for_weight(weight: float, maximum: float) -> tuple[int, int, int]:
    ratio = 0.0 if maximum <= 0 else min(max(weight / maximum, 0.0), 1.0)
    return (255, int(255 * (1.0 - ratio)), 0)


def render_overlay(
    slide_path: Path,
    coordinates: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
    thumbnail_max_size: int,
) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        level_zero_width, level_zero_height = slide.dimensions
        thumbnail = slide.get_thumbnail(
            (thumbnail_max_size, thumbnail_max_size)
        ).convert("RGB")
    finally:
        slide.close()

    scale_x = thumbnail.width / level_zero_width
    scale_y = thumbnail.height / level_zero_height
    draw = ImageDraw.Draw(thumbnail, "RGBA")
    maximum = float(weights.max())
    radius = max(2, int(round(min(thumbnail.size) * 0.006)))

    for (x_value, y_value), weight in zip(coordinates, weights):
        center_x = int(round(float(x_value) * scale_x))
        center_y = int(round(float(y_value) * scale_y))
        red, green, blue = color_for_weight(float(weight), maximum)
        alpha = int(45 + 190 * (float(weight) / maximum)) if maximum > 0 else 45
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=(red, green, blue, alpha),
            outline=(255, 0, 0, min(alpha + 30, 255)),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.save(output_path)


def write_attention_csv(
    path: Path,
    tile_names: list[str],
    coordinates: np.ndarray,
    weights: np.ndarray,
) -> list[dict]:
    order = np.argsort(-weights)
    rows: list[dict] = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "tile_index": int(index),
                "tile_name": tile_names[index],
                "x": int(coordinates[index, 0]),
                "y": int(coordinates[index, 1]),
                "attention_weight": float(weights[index]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def process_slide(
    slide_name: str,
    model: AttentionMIL,
    device: torch.device,
    output_directory: Path,
    top_k: int,
    thumbnail_max_size: int,
) -> dict:
    embedding_directory = PROJECT_ROOT / "embeddings" / "camelyon16"
    embeddings_path = embedding_directory / f"{slide_name}_embeddings.npy"
    coordinates_path = embedding_directory / f"{slide_name}_coordinates.npy"
    names_path = embedding_directory / f"{slide_name}_tile_names.json"
    embeddings = np.load(embeddings_path, allow_pickle=False)
    coordinates = np.load(coordinates_path, allow_pickle=False)
    tile_names = json.loads(names_path.read_text(encoding="utf-8"))

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(f"{slide_name}: unexpected embeddings shape {embeddings.shape}")
    if coordinates.shape != (len(embeddings), 2):
        raise RuntimeError(f"{slide_name}: coordinate alignment failed.")
    if len(tile_names) != len(embeddings):
        raise RuntimeError(f"{slide_name}: tile-name alignment failed.")

    probability, weights = score_slide_bag(model, embeddings, device)
    csv_path = output_directory / f"{slide_name}_attention.csv"
    overlay_path = output_directory / f"{slide_name}_attention_overlay.png"
    rows = write_attention_csv(csv_path, tile_names, coordinates, weights)
    render_overlay(
        find_slide_path(slide_name),
        coordinates,
        weights,
        overlay_path,
        thumbnail_max_size,
    )

    top_rows = rows[: min(top_k, len(rows))]
    return {
        "slide": slide_name,
        "tile_count": int(len(embeddings)),
        "exploratory_probability": probability,
        "attention_sum": float(weights.sum()),
        "attention_minimum": float(weights.min()),
        "attention_maximum": float(weights.max()),
        "attention_entropy": attention_entropy(weights),
        "uniform_attention_entropy": float(math.log(len(weights))),
        "effective_attended_tile_count": float(math.exp(attention_entropy(weights))),
        "attention_csv": str(csv_path),
        "attention_overlay": str(overlay_path),
        "top_tiles": top_rows,
        "passed": True,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.training_seed)
    output_directory = resolve_project_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_bags, training_labels = load_controlled_training_bags(args.training_seed)
    model, epoch_losses = train_attention_model(
        training_bags,
        training_labels,
        args.training_seed,
        args.epochs,
        args.learning_rate,
        args.hidden_dim,
        device,
    )
    model_path = output_directory / "attention_model_state.pt"
    torch.save(model.state_dict(), model_path)

    slide_results = [
        process_slide(
            slide_name,
            model,
            device,
            output_directory,
            args.top_k,
            args.thumbnail_max_size,
        )
        for slide_name in args.slides
    ]

    manifest = {
        "schema_version": "1.0",
        "dataset": "CAMELYON16",
        "scientific_scope": (
            "real slide-bag attention-map mechanics pilot using an Attention MIL "
            "model trained on controlled synthetic PCAM bags; not a real-WSI "
            "performance benchmark"
        ),
        "training_source": "controlled synthetic PCAM bags",
        "training_seed": int(args.training_seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "device": str(device),
        "training_loss_initial": float(epoch_losses[0]),
        "training_loss_final": float(epoch_losses[-1]),
        "model_state_path": str(model_path),
        "slides": slide_results,
        "passed": all(row["passed"] for row in slide_results),
        "warning": (
            "Exploratory probabilities and attention weights must not be interpreted "
            "as validated real-WSI predictions or lesion annotations."
        ),
    }
    manifest_path = output_directory / "attention_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print(json.dumps(manifest, indent=2))
    print()
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
