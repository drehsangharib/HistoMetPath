"""Audit repeated training-only uncertainty and disagreement for two frozen candidates.

This diagnostic compares Spatial v2 mean-pooling logistic regression with the
secondary dual-view mean-concatenation logistic regression using the existing
10-repeat out-of-fold training predictions. It does not refit models, load
validation/test embeddings, or generate validation/test probabilities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from core.wsi.run_camelyon16_batch_pipeline import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Spatial v2 versus concatenation disagreement."
    )
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_v2_concat_disagreement_audit.yaml",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize_probabilities(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))

    prediction_path = project_path(config["training_predictions"])
    stability_path = project_path(config["training_stability_results"])
    validation_path = project_path(config["frozen_validation_results"])
    final_lock_path = project_path(config["final_test_lock"])

    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    if stability["validation_slides_loaded"] != 0 or stability["test_slides_loaded"] != 0:
        raise RuntimeError("Training stability boundary is invalid.")
    if stability["validation_model_outputs_generated"] is not False:
        raise RuntimeError("Training audit generated validation outputs.")
    if stability["test_model_outputs_generated"] is not False:
        raise RuntimeError("Training audit generated test outputs.")
    if validation["test_slides_loaded"] != 0:
        raise RuntimeError("Frozen validation result loaded test slides.")
    if validation["test_model_outputs_generated"] is not False:
        raise RuntimeError("Frozen validation result generated test outputs.")

    primary = config["primary_model"]
    secondary = config["secondary_model"]
    threshold = float(config["decision_threshold"])
    lower = float(config["uncertainty_band_lower"])
    upper = float(config["uncertainty_band_upper"])
    expected_repeats = int(config["expected_repeats"])

    raw = load_csv(prediction_path)
    selected = [row for row in raw if row["model_name"] in {primary, secondary}]
    expected_rows = int(config["expected_training_slides"]) * expected_repeats * 2
    if len(selected) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selected prediction rows; found {len(selected)}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["model_name"], row["slide"])].append(row)

    slide_names = sorted({row["slide"] for row in selected})
    if len(slide_names) != int(config["expected_training_slides"]):
        raise RuntimeError("Unexpected number of training slides.")

    slide_rows = []
    repeat_rows = []
    for slide in slide_names:
        primary_rows = sorted(
            grouped[(primary, slide)], key=lambda row: int(row["repeat"])
        )
        secondary_rows = sorted(
            grouped[(secondary, slide)], key=lambda row: int(row["repeat"])
        )
        if len(primary_rows) != expected_repeats or len(secondary_rows) != expected_repeats:
            raise RuntimeError(f"{slide}: incomplete repeat coverage")

        primary_probabilities = [float(row["probability"]) for row in primary_rows]
        secondary_probabilities = [float(row["probability"]) for row in secondary_rows]
        primary_summary = summarize_probabilities(primary_probabilities)
        secondary_summary = summarize_probabilities(secondary_probabilities)
        label = int(primary_rows[0]["label"])

        sign_disagreement_count = 0
        primary_correct_count = 0
        secondary_correct_count = 0
        absolute_differences = []
        for primary_row, secondary_row in zip(primary_rows, secondary_rows):
            repeat_index = int(primary_row["repeat"])
            if repeat_index != int(secondary_row["repeat"]):
                raise RuntimeError(f"{slide}: repeat alignment mismatch")
            primary_probability = float(primary_row["probability"])
            secondary_probability = float(secondary_row["probability"])
            primary_prediction = int(primary_probability >= threshold)
            secondary_prediction = int(secondary_probability >= threshold)
            disagreement = primary_prediction != secondary_prediction
            sign_disagreement_count += int(disagreement)
            primary_correct_count += int(primary_prediction == label)
            secondary_correct_count += int(secondary_prediction == label)
            difference = secondary_probability - primary_probability
            absolute_difference = abs(difference)
            absolute_differences.append(absolute_difference)
            repeat_rows.append(
                {
                    "slide": slide,
                    "label": label,
                    "repeat": repeat_index,
                    "primary_probability": primary_probability,
                    "secondary_probability": secondary_probability,
                    "secondary_minus_primary": difference,
                    "absolute_probability_difference": absolute_difference,
                    "primary_prediction": primary_prediction,
                    "secondary_prediction": secondary_prediction,
                    "prediction_disagreement": disagreement,
                    "primary_correct": primary_prediction == label,
                    "secondary_correct": secondary_prediction == label,
                }
            )

        primary_mean_prediction = int(primary_summary["mean"] >= threshold)
        secondary_mean_prediction = int(secondary_summary["mean"] >= threshold)
        slide_rows.append(
            {
                "slide": slide,
                "label": label,
                "primary_mean_probability": primary_summary["mean"],
                "primary_probability_std": primary_summary["std"],
                "primary_probability_range": primary_summary["range"],
                "secondary_mean_probability": secondary_summary["mean"],
                "secondary_probability_std": secondary_summary["std"],
                "secondary_probability_range": secondary_summary["range"],
                "mean_absolute_probability_difference": float(np.mean(absolute_differences)),
                "maximum_absolute_probability_difference": float(np.max(absolute_differences)),
                "prediction_disagreement_count": sign_disagreement_count,
                "prediction_disagreement_fraction": sign_disagreement_count / expected_repeats,
                "primary_correct_count": primary_correct_count,
                "secondary_correct_count": secondary_correct_count,
                "primary_mean_prediction": primary_mean_prediction,
                "secondary_mean_prediction": secondary_mean_prediction,
                "mean_prediction_disagreement": primary_mean_prediction != secondary_mean_prediction,
                "primary_mean_uncertain": lower <= primary_summary["mean"] <= upper,
                "secondary_mean_uncertain": lower <= secondary_summary["mean"] <= upper,
                "primary_high_variability": primary_summary["std"] >= float(config["high_variability_std_threshold"]),
                "secondary_high_variability": secondary_summary["std"] >= float(config["high_variability_std_threshold"]),
            }
        )

    all_primary = np.asarray([float(row["primary_probability"]) for row in repeat_rows])
    all_secondary = np.asarray([float(row["secondary_probability"]) for row in repeat_rows])
    correlation = float(np.corrcoef(all_primary, all_secondary)[0, 1])
    disagreements = sum(bool(row["prediction_disagreement"]) for row in repeat_rows)

    summary = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "scientific_scope": config["scientific_scope"],
        "primary_model": primary,
        "secondary_model": secondary,
        "training_slides": len(slide_names),
        "repeats": expected_repeats,
        "paired_prediction_rows": len(repeat_rows),
        "validation_slides_loaded": 0,
        "test_slides_loaded": 0,
        "model_outputs_generated": False,
        "validation_model_outputs_generated": False,
        "test_model_outputs_generated": False,
        "decision_threshold": threshold,
        "uncertainty_band": [lower, upper],
        "prediction_correlation": correlation,
        "mean_absolute_probability_difference": float(np.mean(np.abs(all_secondary - all_primary))),
        "maximum_absolute_probability_difference": float(np.max(np.abs(all_secondary - all_primary))),
        "prediction_disagreement_count": int(disagreements),
        "prediction_disagreement_fraction": float(disagreements / len(repeat_rows)),
        "slides_with_any_prediction_disagreement": int(sum(row["prediction_disagreement_count"] > 0 for row in slide_rows)),
        "slides_with_mean_prediction_disagreement": int(sum(row["mean_prediction_disagreement"] for row in slide_rows)),
        "training_predictions_sha256": sha256_file(prediction_path),
        "training_stability_results_sha256": sha256_file(stability_path),
        "frozen_validation_results_sha256": sha256_file(validation_path),
        "final_test_lock_sha256": sha256_file(final_lock_path),
        "config_sha256": sha256_file(config_path),
        "slides": slide_rows,
        "passed": True,
    }

    output_root = project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "v2_concat_disagreement_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "v2_concat_slide_uncertainty.csv", slide_rows)
    write_csv(output_root / "v2_concat_repeat_disagreement.csv", repeat_rows)

    concise = {key: value for key, value in summary.items() if key != "slides"}
    print(json.dumps(concise, indent=2))
    print("PASS: Training-only v2-versus-concatenation disagreement audit completed.")


if __name__ == "__main__":
    main()
