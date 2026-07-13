"""Config-driven HistoMetPath pseudo-slide MIL runner.

The configuration file uses JSON syntax saved with a ``.yaml`` extension. JSON is
valid YAML 1.2, so this keeps the runner dependency-free while remaining compatible
with standard YAML tooling.

This runner evaluates synthetic PCAM pseudo-slide embeddings. It must not be used to
claim native whole-slide-image performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RunPaths:
    embeddings: Path
    labels: Path
    output_dir: Path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open('r', encoding='utf-8') as handle:
        config = json.load(handle)
    required = {'data', 'split', 'model', 'evaluation', 'output'}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    config['_config_path'] = str(config_path)
    return config


def resolve_paths(config: dict[str, Any], project_root: Path) -> RunPaths:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (project_root / path).resolve()
    return RunPaths(
        embeddings=resolve(config['data']['slide_embeddings']),
        labels=resolve(config['data']['slide_labels']),
        output_dir=resolve(config['output']['directory']),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_bags(paths: RunPaths) -> tuple[list[np.ndarray], np.ndarray]:
    if not paths.embeddings.is_file():
        raise FileNotFoundError(f"Missing pseudo-slide embeddings: {paths.embeddings}")
    if not paths.labels.is_file():
        raise FileNotFoundError(f"Missing pseudo-slide labels: {paths.labels}")
    raw_bags = np.load(paths.embeddings, allow_pickle=True)
    labels = np.asarray(np.load(paths.labels, allow_pickle=False), dtype=np.int64).reshape(-1)
    bags = [np.asarray(bag, dtype=np.float32) for bag in raw_bags]
    if len(bags) != len(labels):
        raise ValueError('Bag and label counts do not match.')
    if len(bags) < 4:
        raise ValueError('At least four pseudo-slides are required.')
    dimensions = {bag.shape[1] for bag in bags if bag.ndim == 2 and bag.shape[0] > 0}
    if len(dimensions) != 1 or any(bag.ndim != 2 or bag.shape[0] == 0 for bag in bags):
        raise ValueError('Every bag must be a non-empty 2D array with one embedding dimension.')
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('Labels must be binary (0 or 1).')
    return bags, labels


def pool_bags(bags: list[np.ndarray], mode: str) -> np.ndarray:
    if mode == 'mean':
        return np.stack([bag.mean(axis=0) for bag in bags])
    if mode == 'max':
        return np.stack([bag.max(axis=0) for bag in bags])
    raise ValueError(f"Unsupported pooling mode: {mode}")


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        score = balanced_accuracy_score(labels, predictions)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {
        'threshold': float(threshold),
        'auroc': safe_auc(labels, probabilities),
        'auprc': float(average_precision_score(labels, probabilities)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall_sensitivity': float(recall_score(labels, predictions, zero_division=0)),
        'specificity': specificity,
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'brier_score': float(brier_score_loss(labels, probabilities)),
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'n_samples': int(len(labels)),
    }


def bootstrap_auc(labels: np.ndarray, probabilities: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            values.append(float(roc_auc_score(sampled_labels, probabilities[indices])))
    if not values:
        return {'lower': None, 'upper': None, 'valid_resamples': 0}
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {'lower': float(lower), 'upper': float(upper), 'valid_resamples': len(values)}


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'mil_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    metrics = summary['metrics']
    with (output_dir / 'mil_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                writer.writerow([key, value])


def run(config_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    seed = int(config.get('seed', 42))
    seed_everything(seed)
    paths = resolve_paths(config, root)
    bags, labels = load_bags(paths)
    indices = np.arange(len(labels))
    test_size = float(config['split'].get('validation_fraction', 0.30))
    stratify = labels if np.unique(labels).size == 2 and np.min(np.bincount(labels)) >= 2 else None
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
    mode = str(config['model'].get('pooling', 'mean')).lower()
    train_features = pool_bags([bags[index] for index in train_idx], mode)
    val_features = pool_bags([bags[index] for index in val_idx], mode)
    classifier = LogisticRegression(max_iter=int(config['model'].get('max_iter', 1000)), random_state=seed)
    classifier.fit(train_features, labels[train_idx])
    train_prob = classifier.predict_proba(train_features)[:, 1]
    val_prob = classifier.predict_proba(val_features)[:, 1]
    threshold = select_threshold(labels[train_idx], train_prob)
    metrics = compute_metrics(labels[val_idx], val_prob, threshold)
    bootstrap_samples = int(config['evaluation'].get('bootstrap_samples', 200))
    metrics['auroc_95_ci'] = bootstrap_auc(labels[val_idx], val_prob, bootstrap_samples, seed)
    summary = {
        'schema_version': '1.0',
        'scientific_scope': 'synthetic PCAM pseudo-slide MIL; not native WSI validation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': config['_config_path'],
        'seed': seed,
        'pooling': mode,
        'train_bags': int(len(train_idx)),
        'validation_bags': int(len(val_idx)),
        'embedding_dimension': int(train_features.shape[1]),
        'metrics': metrics,
    }
    write_outputs(paths.output_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run config-driven HistoMetPath pseudo-slide MIL.')
    parser.add_argument('--config', required=True, help='Path to JSON-compatible YAML configuration.')
    parser.add_argument('--project-root', default=None, help='Optional repository root override.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.config, args.project_root)
    print('HistoMetPath pseudo-slide MIL completed.')
    print(f"Pooling: {summary['pooling']}")
    print(f"Validation bags: {summary['validation_bags']}")
    print('Scientific scope: synthetic pseudo-slides, not native WSI validation.')


if __name__ == '__main__':
    main()
