"""
Multi-seed evaluation for controlled synthetic PCAM pseudo-slides.

Protocol
--------
- Training bags are used to fit the classifier.
- Validation bags are used to select the decision threshold.
- Test bags are evaluated using the frozen classifier and threshold.
- Mean and max pooling are compared.
- Test AUROC confidence intervals are estimated by bootstrap resampling.

The experiment evaluates controlled synthetic PCAM pseudo-slides. It does
not constitute native WSI or patient-level validation.
"""

from __future__ import annotations

import argparse
import csv
import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEEDS = (
    11,
    23,
    42,
    71,
    101,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate controlled synthetic pseudo-slide "
            "MIL across deterministic seeds."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    parser.add_argument(
        "--pooling",
        nargs="+",
        choices=["mean", "max"],
        default=["mean", "max"],
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/mil_controlled",
    )

    return parser.parse_args()


def load_split(
    base_directory: Path,
    split: str,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    split_directory = (
        base_directory
        / split
        / f"seed_{seed}"
    )

    bags_path = split_directory / "bags.npy"
    labels_path = split_directory / "labels.npy"

    if not bags_path.is_file():
        raise FileNotFoundError(bags_path)

    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    raw_bags = np.load(
        bags_path,
        allow_pickle=True,
    )

    labels = np.asarray(
        np.load(
            labels_path,
            allow_pickle=False,
        ),
        dtype=np.int64,
    ).reshape(-1)

    bags = [
        np.asarray(
            bag,
            dtype=np.float32,
        )
        for bag in raw_bags
    ]

    if len(bags) != len(labels):
        raise RuntimeError(
            f"{split}, seed {seed}: bag and label "
            "counts do not match."
        )

    if not bags:
        raise RuntimeError(
            f"{split}, seed {seed}: no bags found."
        )

    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(
            f"{split}, seed {seed}: both binary "
            "classes are required."
        )

    embedding_dimensions = set()

    for bag in bags:
        if bag.ndim != 2:
            raise RuntimeError(
                f"{split}, seed {seed}: every bag "
                "must be two-dimensional."
            )

        if bag.shape[0] == 0:
            raise RuntimeError(
                f"{split}, seed {seed}: empty bag found."
            )

        if not np.isfinite(bag).all():
            raise RuntimeError(
                f"{split}, seed {seed}: nonfinite "
                "embedding values found."
            )

        embedding_dimensions.add(
            bag.shape[1]
        )

    if len(embedding_dimensions) != 1:
        raise RuntimeError(
            f"{split}, seed {seed}: inconsistent "
            "embedding dimensions."
        )

    return bags, labels


def pool_bags(
    bags: list[np.ndarray],
    pooling: str,
) -> np.ndarray:
    if pooling == "mean":
        features = [
            bag.mean(axis=0)
            for bag in bags
        ]
    elif pooling == "max":
        features = [
            bag.max(axis=0)
            for bag in bags
        ]
    else:
        raise ValueError(
            f"Unsupported pooling method: {pooling}"
        )

    return np.stack(
        features
    ).astype(
        np.float32,
        copy=False,
    )


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                probabilities,
                np.array([1.0]),
            ]
        )
    )

    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(
            np.int64
        )

        score = balanced_accuracy_score(
            labels,
            predictions,
        )

        closer_to_half = (
            abs(float(threshold) - 0.5)
            < abs(best_threshold - 0.5)
        )

        if (
            score > best_score
            or (
                np.isclose(
                    score,
                    best_score,
                )
                and closer_to_half
            )
        ):
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (
        probabilities >= threshold
    ).astype(
        np.int64
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(value)
        for value
        in matrix.ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else None
    )

    return {
        "threshold": float(threshold),
        "auroc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "auprc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": (
            float(specificity)
            if specificity is not None
            else None
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
        "sample_count": int(
            len(labels)
        ),
    }


def bootstrap_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(sample_count):
        indices = rng.integers(
            low=0,
            high=len(labels),
            size=len(labels),
        )

        sampled_labels = labels[indices]

        if np.unique(sampled_labels).size != 2:
            continue

        sampled_probabilities = (
            probabilities[indices]
        )

        values.append(
            float(
                roc_auc_score(
                    sampled_labels,
                    sampled_probabilities,
                )
            )
        )

    if not values:
        return {
            "lower": None,
            "upper": None,
            "valid_resamples": 0,
        }

    lower, upper = np.percentile(
        values,
        [2.5, 97.5],
    )

    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_resamples": int(
            len(values)
        ),
    }


def evaluate_one_configuration(
    base_directory: Path,
    seed: int,
    pooling: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    train_bags, train_labels = load_split(
        base_directory,
        "train",
        seed,
    )

    valid_bags, valid_labels = load_split(
        base_directory,
        "valid",
        seed,
    )

    test_bags, test_labels = load_split(
        base_directory,
        "test",
        seed,
    )

    train_features = pool_bags(
        train_bags,
        pooling,
    )

    valid_features = pool_bags(
        valid_bags,
        pooling,
    )

    test_features = pool_bags(
        test_bags,
        pooling,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )

    classifier.fit(
        train_features,
        train_labels,
    )

    validation_probabilities = (
        classifier.predict_proba(
            valid_features
        )[:, 1]
    )

    threshold, validation_score = (
        select_threshold(
            valid_labels,
            validation_probabilities,
        )
    )

    test_probabilities = (
        classifier.predict_proba(
            test_features
        )[:, 1]
    )

    validation_metrics = calculate_metrics(
        valid_labels,
        validation_probabilities,
        threshold,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold,
    )

    test_metrics["auroc_95_ci"] = (
        bootstrap_auroc(
            labels=test_labels,
            probabilities=test_probabilities,
            sample_count=bootstrap_samples,
            seed=seed + 10_000,
        )
    )

    return {
        "seed": int(seed),
        "pooling": pooling,
        "scientific_scope": (
            "controlled synthetic PCAM pseudo-slide "
            "benchmark; not native WSI validation"
        ),
        "train_bags": int(
            len(train_labels)
        ),
        "validation_bags": int(
            len(valid_labels)
        ),
        "test_bags": int(
            len(test_labels)
        ),
        "embedding_dimension": int(
            train_features.shape[1]
        ),
        "validation_selected_threshold": (
            float(threshold)
        ),
        "validation_selection_score": (
            float(validation_score)
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "test_metrics": test_metrics,
    }


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    pooling_methods = sorted(
        {
            result["pooling"]
            for result in results
        }
    )

    for pooling in pooling_methods:
        selected = [
            result
            for result in results
            if result["pooling"] == pooling
        ]

        test_aurocs = np.asarray(
            [
                result["test_metrics"]["auroc"]
                for result in selected
            ],
            dtype=np.float64,
        )

        test_auprcs = np.asarray(
            [
                result["test_metrics"]["auprc"]
                for result in selected
            ],
            dtype=np.float64,
        )

        test_balanced_accuracies = np.asarray(
            [
                result["test_metrics"][
                    "balanced_accuracy"
                ]
                for result in selected
            ],
            dtype=np.float64,
        )

        summary[pooling] = {
            "run_count": int(
                len(selected)
            ),
            "test_auroc_mean": float(
                test_aurocs.mean()
            ),
            "test_auroc_std": float(
                test_aurocs.std(
                    ddof=1
                )
                if len(test_aurocs) > 1
                else 0.0
            ),
            "test_auroc_min": float(
                test_aurocs.min()
            ),
            "test_auroc_max": float(
                test_aurocs.max()
            ),
            "test_auprc_mean": float(
                test_auprcs.mean()
            ),
            "test_auprc_std": float(
                test_auprcs.std(
                    ddof=1
                )
                if len(test_auprcs) > 1
                else 0.0
            ),
            "test_balanced_accuracy_mean": float(
                test_balanced_accuracies.mean()
            ),
            "test_balanced_accuracy_std": float(
                test_balanced_accuracies.std(
                    ddof=1
                )
                if len(
                    test_balanced_accuracies
                ) > 1
                else 0.0
            ),
        }

    return summary


def write_outputs(
    output_directory: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_output = {
        "schema_version": "1.0",
        "scientific_scope": (
            "controlled synthetic PCAM pseudo-slide "
            "benchmark; not native WSI validation"
        ),
        "results": results,
        "cross_seed_summary": summary,
    }

    (
        output_directory
        / "controlled_mil_results.json"
    ).write_text(
        json.dumps(
            complete_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        output_directory
        / "controlled_mil_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "seed",
            "pooling",
            "threshold",
            "validation_auroc",
            "validation_balanced_accuracy",
            "test_auroc",
            "test_auprc",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall_sensitivity",
            "test_specificity",
            "test_f1",
            "test_brier_score",
            "test_auroc_ci_lower",
            "test_auroc_ci_upper",
            "valid_bootstrap_resamples",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            validation = result[
                "validation_metrics"
            ]

            test = result[
                "test_metrics"
            ]

            confidence_interval = test[
                "auroc_95_ci"
            ]

            writer.writerow(
                {
                    "seed": result["seed"],
                    "pooling": result[
                        "pooling"
                    ],
                    "threshold": result[
                        "validation_selected_threshold"
                    ],
                    "validation_auroc": (
                        validation["auroc"]
                    ),
                    "validation_balanced_accuracy": (
                        validation[
                            "balanced_accuracy"
                        ]
                    ),
                    "test_auroc": test["auroc"],
                    "test_auprc": test["auprc"],
                    "test_accuracy": test[
                        "accuracy"
                    ],
                    "test_balanced_accuracy": test[
                        "balanced_accuracy"
                    ],
                    "test_precision": test[
                        "precision"
                    ],
                    "test_recall_sensitivity": test[
                        "recall_sensitivity"
                    ],
                    "test_specificity": test[
                        "specificity"
                    ],
                    "test_f1": test["f1"],
                    "test_brier_score": test[
                        "brier_score"
                    ],
                    "test_auroc_ci_lower": (
                        confidence_interval[
                            "lower"
                        ]
                    ),
                    "test_auroc_ci_upper": (
                        confidence_interval[
                            "upper"
                        ]
                    ),
                    "valid_bootstrap_resamples": (
                        confidence_interval[
                            "valid_resamples"
                        ]
                    ),
                }
            )


def main() -> None:
    args = parse_args()

    base_directory = (
        PROJECT_ROOT
        / "embeddings"
        / "pseudo_slides_controlled"
    )

    output_directory = Path(
        args.output_dir
    )

    if not output_directory.is_absolute():
        output_directory = (
            PROJECT_ROOT / output_directory
        )

    results: list[dict[str, Any]] = []

    for pooling in args.pooling:
        for seed in args.seeds:
            print(
                f"Evaluating pooling={pooling} "
                f"seed={seed}"
            )

            result = evaluate_one_configuration(
                base_directory=base_directory,
                seed=seed,
                pooling=pooling,
                bootstrap_samples=(
                    args.bootstrap_samples
                ),
            )

            results.append(result)

            print(
                "  test AUROC="
                f"{result['test_metrics']['auroc']:.4f} "
                "balanced_accuracy="
                f"{result['test_metrics']['balanced_accuracy']:.4f}"
            )

    summary = summarize_results(results)

    write_outputs(
        output_directory=output_directory,
        results=results,
        summary=summary,
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "PASS: Controlled multi-seed MIL "
        "evaluation completed."
    )

    print(
        f"Outputs: {output_directory}"
    )


if __name__ == "__main__":
    main()
