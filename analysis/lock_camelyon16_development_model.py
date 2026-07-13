"""Train/select CAMELYON16 models using train and validation slides only.

This command refuses to load test embeddings. It compares mean pooling, max
pooling, and a five-seed Attention MIL ensemble, selects one final model using
validation metrics, serializes the selected development artifact, and writes a
checksum lock for a later one-time final-test command.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from analysis.attention_mil_v2 import AttentionMIL
from analysis.run_camelyon16_real_wsi_mil_pilot import select_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"normal": 0, "tumor": 1}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a train/validation-only development lock.")
    parser.add_argument("--config", default="configs/wsi/camelyon16_development_lock.yaml")
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


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    for key in ("processing_manifest", "fresh_holdout_lock", "embedding_root", "output_root", "attention"):
        if key not in config: raise KeyError(f"Missing config key: {key}")
    return config


def verify_holdout_lock(config: dict) -> dict:
    path = project_path(config["fresh_holdout_lock"])
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("test_boundary_status") != "UNTOUCHED_UNTIL_FINAL_EVALUATION":
        raise RuntimeError("Fresh test boundary is not locked.")
    if len(lock.get("fixed_test_slides", [])) != 6:
        raise RuntimeError("Expected six locked test slides.")
    return {"path": str(path), "sha256": sha256_file(path), "test_slides": lock["fixed_test_slides"]}


def load_development_bags(config: dict) -> dict[str, list[dict]]:
    manifest_path = project_path(config["processing_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = project_path(config["embedding_root"])
    grouped = {"train": [], "validation": []}
    for row in manifest["slides"]:
        split = row["split"]
        if split == "test":
            continue
        if split not in grouped:
            raise RuntimeError(f"Unexpected split: {split}")
        path = root / f"{row['slide']}_embeddings.npy"
        bag = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if bag.ndim != 2 or bag.shape[1] != 512 or not np.isfinite(bag).all():
            raise RuntimeError(f"Invalid development bag: {row['slide']}")
        grouped[split].append({
            "slide": row["slide"], "label_text": row["label"],
            "label": LABELS[row["label"]], "bag": np.ascontiguousarray(bag),
            "tile_count": len(bag),
        })
    for split, expected in (("train", 30), ("validation", 6)):
        grouped[split].sort(key=lambda item: item["slide"])
        if len(grouped[split]) != expected: raise RuntimeError(f"Expected {expected} {split} slides.")
        counts = {label: sum(row["label"] == label for row in grouped[split]) for label in (0, 1)}
        required = 15 if split == "train" else 3
        if counts != {0: required, 1: required}: raise RuntimeError(f"Invalid {split} balance: {counts}")
    return grouped


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def pooled(rows: list[dict], mode: str) -> np.ndarray:
    return np.stack([(row["bag"].mean(0) if mode == "mean" else row["bag"].max(0)) for row in rows])


def fit_baseline(grouped: dict, config: dict, mode: str) -> dict:
    train_x, val_x = pooled(grouped["train"], mode), pooled(grouped["validation"], mode)
    train_y = np.array([row["label"] for row in grouped["train"]])
    val_y = np.array([row["label"] for row in grouped["validation"]])
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(C=float(config["baseline"]["logistic_c"]), max_iter=int(config["baseline"]["max_iter"]), random_state=int(config["seed"]))
    classifier.fit(scaler.transform(train_x), train_y)
    probabilities = classifier.predict_proba(scaler.transform(val_x))[:, 1]
    threshold = select_threshold(val_y, probabilities)["threshold"]
    return {
        "model_name": f"{mode}_pooling_logistic_regression",
        "validation_metrics": metrics(val_y, probabilities, threshold),
        "threshold": float(threshold),
        "validation_predictions": [
            {"slide": row["slide"], "label": row["label_text"], "probability": float(prob), "prediction": int(prob >= threshold)}
            for row, prob in zip(grouped["validation"], probabilities)
        ],
        "artifact": {"pooling": mode, "scaler": scaler, "classifier": classifier},
    }


def instance_stats(rows: list[dict]):
    values = np.concatenate([row["bag"] for row in rows], axis=0)
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    std = values.std(0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1.0
    return mean, std


def standardized(rows: list[dict], mean: np.ndarray, std: np.ndarray):
    return [{**row, "bag": np.ascontiguousarray((row["bag"] - mean) / std)} for row in rows]


def predict_attention(model, rows, device):
    output = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            logit, attention = model(torch.from_numpy(row["bag"]).to(device))
            if not torch.isclose(attention.sum(), torch.tensor(1.0, device=device), atol=1e-5): raise RuntimeError("Attention normalization failed.")
            output.append(float(torch.sigmoid(logit).cpu()))
    return np.array(output)


def train_attention(grouped: dict, config: dict, device) -> dict:
    cfg = config["attention"]
    mean, std = instance_stats(grouped["train"])
    train_rows, val_rows = standardized(grouped["train"], mean, std), standardized(grouped["validation"], mean, std)
    val_y = np.array([row["label"] for row in val_rows])
    probabilities_by_seed, states, runs = [], [], []
    for seed in cfg["seeds"]:
        seed_all(int(seed)); rng = np.random.default_rng(int(seed))
        model = AttentionMIL(in_dim=512, hidden_dim=int(cfg["hidden_dim"])).to(device)
        optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), math.inf, 0, 0
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            for index in rng.permutation(len(train_rows)):
                row = train_rows[index]; tensor = torch.from_numpy(row["bag"]).to(device)
                label = torch.tensor(float(row["label"]), device=device)
                logit, _ = model(tensor); loss = F.binary_cross_entropy_with_logits(logit.reshape(1), label.reshape(1))
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"])); optimizer.step()
            model.eval(); losses = []
            with torch.inference_mode():
                for row in val_rows:
                    logit, _ = model(torch.from_numpy(row["bag"]).to(device))
                    label = torch.tensor(float(row["label"]), device=device)
                    losses.append(float(F.binary_cross_entropy_with_logits(logit.reshape(1), label.reshape(1)).cpu()))
            val_loss = float(np.mean(losses))
            if val_loss < best_loss - float(cfg["min_delta"]):
                best_loss, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= int(cfg["patience"]): break
        model.load_state_dict(best_state)
        probabilities_by_seed.append(predict_attention(model, val_rows, device))
        states.append({key: value.cpu() for key, value in best_state.items()})
        runs.append({"seed": int(seed), "best_epoch": best_epoch, "best_validation_loss": best_loss})
        print(f"Attention seed {seed}: best_epoch={best_epoch}, best_validation_loss={best_loss:.6f}")
    probabilities = np.mean(probabilities_by_seed, axis=0)
    threshold = select_threshold(val_y, probabilities)["threshold"]
    return {
        "model_name": "attention_mil_five_seed_probability_ensemble",
        "validation_metrics": metrics(val_y, probabilities, threshold),
        "threshold": float(threshold),
        "validation_predictions": [{"slide": row["slide"], "label": row["label_text"], "probability": float(prob), "prediction": int(prob >= threshold)} for row, prob in zip(val_rows, probabilities)],
        "artifact": {"model_type": "attention_ensemble", "hidden_dim": int(cfg["hidden_dim"]), "seeds": [int(x) for x in cfg["seeds"]], "instance_mean": mean, "instance_std": std, "state_dicts": states},
        "training_runs": runs,
    }


def selection_key(result: dict):
    value = result["validation_metrics"]
    return (value["balanced_accuracy"], value["auroc"], value["auprc"], result["model_name"])


def json_safe(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "artifact"}


def main() -> None:
    args = parse_args(); config_path = project_path(args.config); config = load_config(config_path)
    seed_all(int(config["seed"])); holdout = verify_holdout_lock(config); grouped = load_development_bags(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [fit_baseline(grouped, config, "mean"), fit_baseline(grouped, config, "max"), train_attention(grouped, config, device)]
    selected = max(candidates, key=selection_key)
    output_root = project_path(config["output_root"]); output_root.mkdir(parents=True, exist_ok=True)
    artifact_path = output_root / "selected_development_model.joblib"
    joblib.dump({"model_name": selected["model_name"], "threshold": selected["threshold"], "artifact": selected["artifact"]}, artifact_path)
    report = {
        "schema_version": "1.0", "dataset": config["dataset"], "scientific_scope": config["scientific_scope"],
        "git_commit": None, "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "processing_manifest_sha256": sha256_file(project_path(config["processing_manifest"])),
        "holdout_lock": holdout, "development_counts": {"train": 30, "validation": 6, "test_loaded": 0},
        "candidates": [json_safe(item) for item in candidates], "selected_model": selected["model_name"],
        "selected_threshold": selected["threshold"], "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path), "test_boundary_status": "UNTOUCHED", "passed": True,
    }
    try:
        import subprocess
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        report["git_commit"] = "unavailable"
    report_path = output_root / "development_lock.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": report["selected_model"], "selected_threshold": report["selected_threshold"], "test_loaded": 0, "test_boundary_status": "UNTOUCHED", "artifact_sha256": report["artifact_sha256"]}, indent=2))
    print(f"Development lock written to: {report_path}")
    print("PASS: Development model selected without loading fresh test embeddings.")


if __name__ == "__main__": main()
