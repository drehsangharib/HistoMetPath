"""Validate and summarize controlled Attention MIL results."""

import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT / "outputs" / "mil_controlled_attention" / "attention_results.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "mil_controlled_attention" / "attention_summary.json"
)

EXPECTED_SEEDS = [11, 23, 42, 71, 101]

METRIC_NAMES = [
    "threshold",
    "test_auroc",
    "test_auprc",
    "test_balanced_accuracy",
    "attention_mean_entropy",
]

def metric_values(results, name):
    return [float(r[name]) for r in results]

def describe(values):
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }

def validate_results(results):
    if not isinstance(results, list):
        raise TypeError("Attention results must be a JSON list.")

    if len(results) != len(EXPECTED_SEEDS):
        raise RuntimeError(f"Expected 5 records, found {len(results)}.")

    observed_seeds = sorted(int(r["seed"]) for r in results)
    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"Unexpected seeds: {observed_seeds}")

    for r in results:
        seed = int(r["seed"])

        for m in METRIC_NAMES:
            if m not in r:
                raise KeyError(f"Seed {seed} missing {m}")
            v = float(r[m])
            if not math.isfinite(v):
                raise RuntimeError(f"Seed {seed} nonfinite {m}: {v}")

        for m in ["threshold","test_auroc","test_auprc","test_balanced_accuracy"]:
            v = float(r[m])
            if v < 0.0 or v > 1.0:
                raise RuntimeError(f"Seed {seed} invalid {m}: {v}")

        entropy = float(r["attention_mean_entropy"])
        if entropy < 0.0 or entropy > math.log(50.0):
            raise RuntimeError(f"Seed {seed} invalid entropy: {entropy}")

def main():
    if not RESULT_PATH.is_file():
        raise FileNotFoundError(f"Missing {RESULT_PATH}")

    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    validate_results(results)

    summary = {
        "schema_version": "1.0",
        "scientific_scope": "controlled synthetic PCAM pseudo-slide benchmark; not native WSI or patient-level validation",
        "model": "Attention MIL",
        "run_count": len(results),
        "seeds": EXPECTED_SEEDS,
        "test_auroc": describe(metric_values(results,"test_auroc")),
        "test_auprc": describe(metric_values(results,"test_auprc")),
        "test_balanced_accuracy": describe(metric_values(results,"test_balanced_accuracy")),
        "attention_entropy": {
            **describe(metric_values(results,"attention_mean_entropy")),
            "uniform_attention_entropy": math.log(50.0)
        },
        "validation_selected_threshold": describe(metric_values(results,"threshold")),
        "calibration_warning": "Validation-selected thresholds were highly unstable across seeds.",
        "interpretability_warning": "Attention weights do not imply biological causality.",
        "individual_results": results
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("PASS: Attention MIL summary generated.")
    print(f"Written to: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
