"""Repeated training-only stability audit for five interpretable baselines.

The six frozen validation slides and all consumed test slides are excluded.
Each outer test fold is predicted by models fitted only on the corresponding
outer training fold. Stacking uses inner out-of-fold probabilities generated
inside the outer training fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core.wsi.run_camelyon16_batch_pipeline import project_path

LABELS = {"normal": 0, "tumor": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated training-only dual-view stability audit."
    )
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_training_only_stability.yaml",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_lr(x: np.ndarray, y: np.ndarray, config: dict, seed: int):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=float(config["c"]),
        max_iter=int(config["max_iter"]),
        random_state=seed,
    ).fit(scaler.transform(x), y)
    return scaler, model


def predict_lr(bundle, x: np.ndarray) -> np.ndarray:
    scaler, model = bundle
    return model.predict_proba(scaler.transform(x))[:, 1]


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def load_training_data(config: dict):
    manifest_path = project_path(config["dual_view_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_slides_loaded") != 0:
        raise RuntimeError("Dual-view source manifest loaded test slides.")

    records = manifest["records"]
    grouped: dict[str, dict] = {}
    for record in records:
        if record["split"] != config["allowed_split"]:
            continue
        item = grouped.setdefault(
            record["slide"],
            {
                "slide": record["slide"],
                "label": record["label"],
                "split": record["split"],
            },
        )
        item[record["view"]] = record

    if len(grouped) != 30:
        raise RuntimeError(f"Expected 30 training slides; found {len(grouped)}")

    root = project_path(config["embedding_root"])
    rows = []
    for slide_name in sorted(grouped):
        item = grouped[slide_name]
        if item["split"] in set(config["prohibited_splits"]):
            raise RuntimeError(f"Prohibited split entered stability audit: {slide_name}")
        row = {
            "slide": slide_name,
            "label": LABELS[item["label"]],
            "label_text": item["label"],
        }
        for view in config["views"]:
            if view not in item:
                raise RuntimeError(f"{slide_name}: missing view {view}")
            array = np.load(
                root / view / f"{slide_name}_embeddings.npy",
                allow_pickle=False,
            )
            if array.shape != (300, 512) or not np.isfinite(array).all():
                raise RuntimeError(f"{view}/{slide_name}: invalid embeddings {array.shape}")
            row[view] = array.mean(axis=0).astype(np.float64)
        rows.append(row)

    frozen_validation_path = project_path(config["frozen_validation_result"])
    if not frozen_validation_path.is_file():
        raise FileNotFoundError(frozen_validation_path)
    return rows, manifest_path, frozen_validation_path


def inner_stacking_predictions(
    x_views: dict[str, np.ndarray],
    y: np.ndarray,
    views: list[str],
    folds: int,
    seed: int,
    lr_config: dict,
) -> tuple[np.ndarray, dict]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out_of_fold = np.zeros((len(y), len(views)), dtype=np.float64)
    full_models = {}
    for view_index, view in enumerate(views):
        for inner_train, inner_validation in splitter.split(x_views[view], y):
            bundle = fit_lr(
                x_views[view][inner_train],
                y[inner_train],
                lr_config,
                seed,
            )
            out_of_fold[inner_validation, view_index] = predict_lr(
                bundle,
                x_views[view][inner_validation],
            )
        full_models[view] = fit_lr(x_views[view], y, lr_config, seed)
    return out_of_fold, full_models


def coefficient_vector(bundle) -> np.ndarray:
    scaler, model = bundle
    return (model.coef_[0] / scaler.scale_).astype(np.float64)


def mean_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0
    normalized = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        normalized.append(vector / norm if norm > 0 else vector)
    values = []
    for first in range(len(normalized)):
        for second in range(first + 1, len(normalized)):
            values.append(float(np.dot(normalized[first], normalized[second])))
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    rows, manifest_path, frozen_validation_path = load_training_data(config)

    views = list(config["views"])
    model_names = list(config["models"])
    seeds = [int(value) for value in config["seeds"]]
    repeats = int(config["outer_repeats"])
    folds = int(config["outer_folds"])
    if repeats != len(seeds):
        raise RuntimeError("outer_repeats must equal the number of seeds")

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    slides = np.asarray([row["slide"] for row in rows])
    x_views = {
        view: np.stack([row[view] for row in rows])
        for view in views
    }
    concatenated = np.concatenate([x_views[view] for view in views], axis=1)
    threshold = float(config["decision_threshold"])
    lr_config = config["logistic_regression"]

    fold_rows = []
    prediction_rows = []
    rank_counts = Counter()
    coefficient_vectors = defaultdict(list)

    for repeat_index, seed in enumerate(seeds):
        outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_index, (train_index, test_index) in enumerate(
            outer.split(concatenated, labels)
        ):
            y_train = labels[train_index]
            y_test = labels[test_index]
            probabilities = {}

            base_bundles = {}
            for view in views:
                bundle = fit_lr(
                    x_views[view][train_index],
                    y_train,
                    lr_config,
                    seed,
                )
                base_bundles[view] = bundle
                probabilities[f"{view}_mean_pool_lr"] = predict_lr(
                    bundle,
                    x_views[view][test_index],
                )
                coefficient_vectors[f"{view}_mean_pool_lr"].append(
                    coefficient_vector(bundle)
                )

            concat_bundle = fit_lr(
                concatenated[train_index],
                y_train,
                lr_config,
                seed,
            )
            probabilities["dual_view_mean_concat_lr"] = predict_lr(
                concat_bundle,
                concatenated[test_index],
            )
            coefficient_vectors["dual_view_mean_concat_lr"].append(
                coefficient_vector(concat_bundle)
            )

            probabilities["dual_view_late_probability_average"] = np.mean(
                np.column_stack(
                    [
                        probabilities[f"{view}_mean_pool_lr"]
                        for view in views
                    ]
                ),
                axis=1,
            )

            outer_train_views = {
                view: x_views[view][train_index]
                for view in views
            }
            out_of_fold, full_models = inner_stacking_predictions(
                outer_train_views,
                y_train,
                views,
                int(config["inner_stacking_folds"]),
                seed,
                lr_config,
            )
            meta_bundle = fit_lr(out_of_fold, y_train, lr_config, seed)
            outer_test_meta = np.column_stack(
                [
                    predict_lr(full_models[view], x_views[view][test_index])
                    for view in views
                ]
            )
            probabilities["dual_view_oof_logistic_stacking"] = predict_lr(
                meta_bundle,
                outer_test_meta,
            )
            coefficient_vectors["dual_view_oof_logistic_stacking"].append(
                coefficient_vector(meta_bundle)
            )

            fold_metrics = {}
            for model_name in model_names:
                block = metrics(y_test, probabilities[model_name], threshold)
                fold_metrics[model_name] = block
                fold_rows.append(
                    {
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold_index,
                        "model_name": model_name,
                        **block,
                    }
                )
                for slide, label, probability in zip(
                    slides[test_index],
                    y_test,
                    probabilities[model_name],
                ):
                    prediction_rows.append(
                        {
                            "repeat": repeat_index,
                            "seed": seed,
                            "fold": fold_index,
                            "model_name": model_name,
                            "slide": str(slide),
                            "label": int(label),
                            "probability": float(probability),
                            "prediction": int(probability >= threshold),
                        }
                    )

            ranked = sorted(
                model_names,
                key=lambda name: (
                    fold_metrics[name]["auroc"],
                    fold_metrics[name]["auprc"],
                    fold_metrics[name]["balanced_accuracy"],
                    name,
                ),
                reverse=True,
            )
            rank_counts[ranked[0]] += 1

    aggregate = {}
    for model_name in model_names:
        model_folds = [row for row in fold_rows if row["model_name"] == model_name]
        aggregate[model_name] = {
            f"mean_{metric_name}": float(np.mean([row[metric_name] for row in model_folds]))
            for metric_name in ["balanced_accuracy", "auroc", "auprc", "accuracy", "f1"]
        }
        aggregate[model_name].update(
            {
                f"std_{metric_name}": float(np.std([row[metric_name] for row in model_folds], ddof=1))
                for metric_name in ["balanced_accuracy", "auroc", "auprc", "accuracy", "f1"]
            }
        )
        aggregate[model_name]["fold_win_count"] = int(rank_counts[model_name])
        aggregate[model_name]["fold_win_fraction"] = float(
            rank_counts[model_name] / (repeats * folds)
        )
        vectors = coefficient_vectors.get(model_name, [])
        aggregate[model_name]["mean_pairwise_coefficient_cosine"] = (
            mean_pairwise_cosine(vectors) if vectors else None
        )

    priority = list(config["ranking_priority"])
    selected = sorted(
        model_names,
        key=lambda name: tuple(
            aggregate[name][criterion]
            if criterion != "model_name"
            else name
            for criterion in priority
        ),
        reverse=True,
    )[0]

    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "scientific_scope": config["scientific_scope"],
        "training_slides": 30,
        "validation_slides_loaded": 0,
        "test_slides_loaded": 0,
        "model_outputs_generated": True,
        "validation_model_outputs_generated": False,
        "test_model_outputs_generated": False,
        "outer_repeats": repeats,
        "outer_folds": folds,
        "total_outer_folds": repeats * folds,
        "decision_threshold": threshold,
        "aggregate_metrics": aggregate,
        "training_only_selected_model": selected,
        "dual_view_manifest_sha256": sha256_file(manifest_path),
        "frozen_validation_result_sha256": sha256_file(frozen_validation_path),
        "final_test_lock_sha256": sha256_file(project_path(config["final_test_lock"])),
        "config_sha256": sha256_file(config_path),
        "passed": True,
    }
    (output_root / "training_only_stability_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for filename, data in [
        ("training_only_fold_metrics.csv", fold_rows),
        ("training_only_oof_predictions.csv", prediction_rows),
    ]:
        with (output_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    summary_rows = [
        {"model_name": model_name, **aggregate[model_name]}
        for model_name in model_names
    ]
    with (output_root / "training_only_stability_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    concise = {key: value for key, value in report.items() if key != "aggregate_metrics"}
    concise["aggregate_metrics"] = aggregate
    print(json.dumps(concise, indent=2))
    print("PASS: Training-only repeated stability audit completed.")


if __name__ == "__main__":
    main()
