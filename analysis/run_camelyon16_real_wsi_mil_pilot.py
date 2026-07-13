"""Run the first genuine CAMELYON16 real-WSI MIL development pilot.

Protocol
--------
- Train: 14 real slide bags (7 normal, 7 tumor)
- Validation: 4 real slide bags (2 normal, 2 tumor)
- Test: 4 held-out real slide bags (2 normal, 2 tumor)
- Mean and max pooling use train-fitted feature scaling and logistic regression.
- Attention MIL uses train-instance standardization, validation-loss early stopping,
  and a predeclared five-seed probability ensemble.
- Decision thresholds are selected on validation only and frozen for test.

The four-slide test set is too small for stable performance claims. The script
writes individual predictions and explicit limitations alongside aggregate metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from analysis.attention_mil_v2 import AttentionMIL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = ("train", "validation", "test")
LABEL_TO_INT = {"normal": 0, "tumor": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe real-WSI CAMELYON16 MIL pilot."
    )
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_real_wsi_mil_pilot.yaml",
    )
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
        "processing_manifest",
        "embedding_root",
        "output_root",
        "baseline",
        "attention",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Configuration is missing keys: {missing}")
    return config


def load_slide_bags(config: dict) -> dict[str, list[dict]]:
    manifest_path = project_path(config["processing_manifest"])
    embedding_root = project_path(config["embedding_root"])
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True:
        raise RuntimeError("Batch-processing manifest did not pass.")
    if len(manifest.get("slides", [])) != 22:
        raise RuntimeError("Expected exactly 22 slide records.")

    grouped: dict[str, list[dict]] = {split: [] for split in SPLIT_ORDER}
    seen: set[str] = set()
    for row in manifest["slides"]:
        slide_name = row["slide"]
        split = row["split"]
        label_text = row["label"]
        if slide_name in seen:
            raise RuntimeError(f"Duplicate slide record: {slide_name}")
        seen.add(slide_name)
        if split not in grouped:
            raise ValueError(f"Unexpected split: {split}")
        if label_text not in LABEL_TO_INT:
            raise ValueError(f"Unexpected label: {label_text}")
        embeddings_path = embedding_root / f"{slide_name}_embeddings.npy"
        if not embeddings_path.is_file():
            raise FileNotFoundError(embeddings_path)
        bag = np.load(embeddings_path, allow_pickle=False).astype(np.float32, copy=False)
        if bag.ndim != 2 or bag.shape[1] != 512:
            raise RuntimeError(f"{slide_name}: invalid bag shape {bag.shape}")
        if not np.isfinite(bag).all():
            raise RuntimeError(f"{slide_name}: nonfinite embeddings")
        if int(row["tile_count"]) != len(bag):
            raise RuntimeError(f"{slide_name}: manifest/bag count mismatch")
        grouped[split].append(
            {
                "slide": slide_name,
                "label_text": label_text,
                "label": LABEL_TO_INT[label_text],
                "bag": np.ascontiguousarray(bag),
                "tile_count": int(len(bag)),
            }
        )

    expected = {
        "train": {0: 7, 1: 7},
        "validation": {0: 2, 1: 2},
        "test": {0: 2, 1: 2},
    }
    for split, rows in grouped.items():
        observed = {
            label: sum(row["label"] == label for row in rows)
            for label in (0, 1)
        }
        if observed != expected[split]:
            raise RuntimeError(
                f"Unexpected {split} class counts: {observed}; expected {expected[split]}"
            )
        rows.sort(key=lambda row: row["slide"])
    return grouped


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = np.unique(
        np.concatenate(([0.0], probabilities.astype(float), [1.0]))
    )
    best_threshold = 0.5
    best_score = -1.0
    for candidate in candidates:
        predictions = (probabilities >= candidate).astype(int)
        score = balanced_accuracy_score(labels, predictions)
        better_tie = (
            np.isclose(score, best_score)
            and abs(float(candidate) - 0.5) < abs(best_threshold - 0.5)
        )
        if score > best_score or better_tie:
            best_score = float(score)
            best_threshold = float(candidate)
    return {
        "threshold": best_threshold,
        "validation_balanced_accuracy": best_score,
        "candidate_count": int(len(candidates)),
    }


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {
        "sample_count": int(len(labels)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def prediction_rows(
    rows: list[dict],
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> list[dict]:
    output = []
    for row, probability in zip(rows, probabilities):
        output.append(
            {
                "model": model_name,
                "slide": row["slide"],
                "label": row["label_text"],
                "label_binary": int(row["label"]),
                "tile_count": int(row["tile_count"]),
                "probability": float(probability),
                "threshold": float(threshold),
                "prediction_binary": int(probability >= threshold),
                "prediction": "tumor" if probability >= threshold else "normal",
                "correct": bool(int(probability >= threshold) == row["label"]),
            }
        )
    return output


def pooled_features(rows: list[dict], pooling: str) -> np.ndarray:
    operation: Callable[[np.ndarray], np.ndarray]
    if pooling == "mean":
        operation = lambda bag: bag.mean(axis=0)
    elif pooling == "max":
        operation = lambda bag: bag.max(axis=0)
    else:
        raise ValueError(pooling)
    return np.stack([operation(row["bag"]) for row in rows]).astype(np.float64)


def evaluate_baseline(
    grouped: dict[str, list[dict]],
    pooling: str,
    logistic_c: float,
    max_iter: int,
    seed: int,
) -> dict:
    train_rows = grouped["train"]
    validation_rows = grouped["validation"]
    test_rows = grouped["test"]
    train_x = pooled_features(train_rows, pooling)
    validation_x = pooled_features(validation_rows, pooling)
    test_x = pooled_features(test_rows, pooling)
    train_y = np.asarray([row["label"] for row in train_rows], dtype=int)
    validation_y = np.asarray([row["label"] for row in validation_rows], dtype=int)
    test_y = np.asarray([row["label"] for row in test_rows], dtype=int)

    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        C=logistic_c,
        max_iter=max_iter,
        random_state=seed,
        class_weight=None,
    )
    classifier.fit(scaler.transform(train_x), train_y)
    validation_probabilities = classifier.predict_proba(
        scaler.transform(validation_x)
    )[:, 1]
    threshold_result = select_threshold(validation_y, validation_probabilities)
    threshold = threshold_result["threshold"]
    test_probabilities = classifier.predict_proba(scaler.transform(test_x))[:, 1]

    return {
        "model": f"{pooling}_pooling_logistic_regression",
        "training_slide_count": len(train_rows),
        "validation_slide_count": len(validation_rows),
        "test_slide_count": len(test_rows),
        "logistic_c": float(logistic_c),
        "threshold_selection": threshold_result,
        "validation_metrics": calculate_metrics(
            validation_y, validation_probabilities, threshold
        ),
        "test_metrics": calculate_metrics(test_y, test_probabilities, threshold),
        "validation_predictions": prediction_rows(
            validation_rows,
            validation_probabilities,
            threshold,
            f"{pooling}_pooling",
        ),
        "test_predictions": prediction_rows(
            test_rows,
            test_probabilities,
            threshold,
            f"{pooling}_pooling",
        ),
    }


def fit_instance_scaler(train_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_instances = np.concatenate([row["bag"] for row in train_rows], axis=0)
    mean = all_instances.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = all_instances.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_rows(
    rows: list[dict], mean: np.ndarray, std: np.ndarray
) -> list[dict]:
    output = []
    for row in rows:
        copied = dict(row)
        copied["bag"] = np.ascontiguousarray((row["bag"] - mean) / std)
        output.append(copied)
    return output


def predict_attention(
    model: AttentionMIL,
    rows: list[dict],
    device: torch.device,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            tensor = torch.from_numpy(row["bag"]).to(device)
            logit, attention = model(tensor)
            if not torch.isfinite(logit):
                raise RuntimeError("Nonfinite attention-model logit.")
            if not torch.isclose(attention.sum(), torch.tensor(1.0, device=device), atol=1e-5):
                raise RuntimeError("Attention weights do not sum to one.")
            probabilities.append(float(torch.sigmoid(logit).cpu()))
    return np.asarray(probabilities, dtype=np.float64)


def train_attention_seed(
    train_rows: list[dict],
    validation_rows: list[dict],
    config: dict,
    seed: int,
    device: torch.device,
) -> dict:
    seed_everything(seed)
    model = AttentionMIL(in_dim=512, hidden_dim=int(config["hidden_dim"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        train_losses = []
        for index in rng.permutation(len(train_rows)):
            row = train_rows[index]
            tensor = torch.from_numpy(row["bag"]).to(device)
            label = torch.tensor(float(row["label"]), device=device)
            logit, _ = model(tensor)
            loss = F.binary_cross_entropy_with_logits(logit.reshape(1), label.reshape(1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for row in validation_rows:
                tensor = torch.from_numpy(row["bag"]).to(device)
                label = torch.tensor(float(row["label"]), device=device)
                logit, _ = model(tensor)
                loss = F.binary_cross_entropy_with_logits(
                    logit.reshape(1), label.reshape(1)
                )
                validation_losses.append(float(loss.cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Attention seed={seed} epoch={epoch}: "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss - float(config["min_delta"]):
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["patience"]):
                break

    model.load_state_dict(best_state)
    return {
        "seed": seed,
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def evaluate_attention_ensemble(
    grouped: dict[str, list[dict]],
    attention_config: dict,
    device: torch.device,
) -> dict:
    train_mean, train_std = fit_instance_scaler(grouped["train"])
    standardized = {
        split: standardize_rows(rows, train_mean, train_std)
        for split, rows in grouped.items()
    }
    validation_y = np.asarray(
        [row["label"] for row in standardized["validation"]], dtype=int
    )
    test_y = np.asarray([row["label"] for row in standardized["test"]], dtype=int)
    validation_seed_probabilities = []
    test_seed_probabilities = []
    training_runs = []

    for seed in attention_config["seeds"]:
        result = train_attention_seed(
            standardized["train"],
            standardized["validation"],
            attention_config,
            int(seed),
            device,
        )
        model = result.pop("model")
        validation_probabilities = predict_attention(
            model, standardized["validation"], device
        )
        test_probabilities = predict_attention(model, standardized["test"], device)
        validation_seed_probabilities.append(validation_probabilities)
        test_seed_probabilities.append(test_probabilities)
        training_runs.append(
            {
                **result,
                "validation_probabilities": validation_probabilities.tolist(),
                "test_probabilities": test_probabilities.tolist(),
            }
        )

    validation_ensemble = np.mean(validation_seed_probabilities, axis=0)
    test_ensemble = np.mean(test_seed_probabilities, axis=0)
    threshold_result = select_threshold(validation_y, validation_ensemble)
    threshold = threshold_result["threshold"]

    return {
        "model": "attention_mil_five_seed_probability_ensemble",
        "ensemble_seeds": [int(seed) for seed in attention_config["seeds"]],
        "training_slide_count": len(standardized["train"]),
        "validation_slide_count": len(standardized["validation"]),
        "test_slide_count": len(standardized["test"]),
        "threshold_selection": threshold_result,
        "validation_metrics": calculate_metrics(
            validation_y, validation_ensemble, threshold
        ),
        "test_metrics": calculate_metrics(test_y, test_ensemble, threshold),
        "validation_predictions": prediction_rows(
            standardized["validation"],
            validation_ensemble,
            threshold,
            "attention_ensemble",
        ),
        "test_predictions": prediction_rows(
            standardized["test"],
            test_ensemble,
            threshold,
            "attention_ensemble",
        ),
        "training_runs": training_runs,
        "standardization": {
            "fit_scope": "all tile instances from training slides only",
            "feature_count": int(len(train_mean)),
        },
    }


def write_prediction_csv(path: Path, results: list[dict]) -> None:
    rows = []
    for result in results:
        for split_name in ("validation", "test"):
            for row in result[f"{split_name}_predictions"]:
                rows.append({"split": split_name, **row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    grouped = load_slide_bags(config)
    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        evaluate_baseline(
            grouped,
            pooling="mean",
            logistic_c=float(config["baseline"]["logistic_c"]),
            max_iter=int(config["baseline"]["max_iter"]),
            seed=int(config.get("seed", 42)),
        ),
        evaluate_baseline(
            grouped,
            pooling="max",
            logistic_c=float(config["baseline"]["logistic_c"]),
            max_iter=int(config["baseline"]["max_iter"]),
            seed=int(config.get("seed", 42)),
        ),
        evaluate_attention_ensemble(grouped, config["attention"], device),
    ]

    output = {
        "schema_version": "1.0",
        "dataset": config.get("dataset", "CAMELYON16"),
        "scientific_scope": config.get("scientific_scope"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "processing_manifest_path": str(project_path(config["processing_manifest"])),
        "processing_manifest_sha256": sha256_file(
            project_path(config["processing_manifest"])
        ),
        "device": str(device),
        "split_counts": {
            split: {
                "total": len(rows),
                "normal": sum(row["label"] == 0 for row in rows),
                "tumor": sum(row["label"] == 1 for row in rows),
            }
            for split, rows in grouped.items()
        },
        "results": results,
        "limitations": [
            "The held-out test set contains only four slides.",
            "Aggregate test metrics are highly unstable at this sample size.",
            "The patch encoder was originally trained on PCAM patches.",
            "The first 300 accepted tissue tiles are sampled deterministically and may miss small lesions.",
            "The results are a development pilot and not evidence of clinical utility.",
        ],
        "passed": True,
    }
    results_path = output_root / "real_wsi_mil_results.json"
    predictions_path = output_root / "real_wsi_mil_predictions.csv"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_prediction_csv(predictions_path, results)

    summary = {
        result["model"]: {
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
        }
        for result in results
    }
    (output_root / "real_wsi_mil_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print()
    print(f"Results written to: {results_path}")
    print(f"Predictions written to: {predictions_path}")
    print("PASS: Real-WSI MIL development pilot completed.")


if __name__ == "__main__":
    main()
