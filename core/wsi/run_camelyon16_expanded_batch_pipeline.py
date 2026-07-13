"""Process a 42-slide CAMELYON16 expansion with predeclared fresh holdouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from core.wsi.run_camelyon16_batch_pipeline import (
    PROJECT_ROOT,
    discover_slides,
    load_config,
    load_model,
    process_slide,
    project_path,
    seed_everything,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded CAMELYON16 batch pipeline.")
    parser.add_argument(
        "--config",
        default="configs/wsi/camelyon16_expanded_fresh_holdout.yaml",
    )
    parser.add_argument(
        "--stage", choices=["registry", "process", "all"], default="all"
    )
    parser.add_argument("--slides", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def assign_fixed_splits(slides: list[dict], config: dict) -> list[dict]:
    validation = set(config["fixed_validation_slides"])
    test = set(config["fixed_test_slides"])
    if validation & test:
        raise RuntimeError("Validation and test slide sets overlap.")
    discovered = {row["slide"] for row in slides}
    missing = sorted((validation | test) - discovered)
    if missing:
        raise RuntimeError(f"Predeclared holdout slides are missing: {missing}")

    assigned = []
    for source in slides:
        row = dict(source)
        if row["slide"] in test:
            row["split"] = "test"
        elif row["slide"] in validation:
            row["split"] = "validation"
        else:
            row["split"] = "train"
        assigned.append(row)
    assigned.sort(key=lambda row: row["slide"])

    expected = config["expected_counts_per_class"]
    for split_name in ("train", "validation", "test"):
        for label in ("normal", "tumor"):
            observed = sum(
                row["split"] == split_name and row["label"] == label
                for row in assigned
            )
            required = int(expected[split_name])
            if observed != required:
                raise RuntimeError(
                    f"Expected {required} {label} slides in {split_name}; found {observed}."
                )
    return assigned


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_registry(config: dict, output_root: Path) -> list[dict]:
    slides = discover_slides(project_path(config["data_root"]))
    if len(slides) != 42:
        raise RuntimeError(f"Expected 42 local slides; found {len(slides)}.")
    registry = assign_fixed_splits(slides, config)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "slide_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    write_csv(output_root / "slide_registry.csv", registry)
    summary = {
        split_name: {
            label: sum(
                row["split"] == split_name and row["label"] == label
                for row in registry
            )
            for label in ("normal", "tumor")
        }
        for split_name in ("train", "validation", "test")
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lock = {
        "schema_version": "1.0",
        "scientific_scope": config["scientific_scope"],
        "fixed_validation_slides": sorted(config["fixed_validation_slides"]),
        "fixed_test_slides": sorted(config["fixed_test_slides"]),
        "test_boundary_status": "UNTOUCHED_UNTIL_FINAL_EVALUATION",
    }
    (output_root / "fresh_holdout_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    return registry


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    output_root = project_path(config["output_root"])
    embedding_root = project_path(config["embedding_root"])
    registry = build_registry(config, output_root)

    if args.stage == "registry":
        print(json.dumps({"slides": 42, "stage": "registry", "passed": True}, indent=2))
        return

    selected = registry
    if args.slides:
        requested = set(args.slides)
        selected = [row for row in registry if row["slide"] in requested]
        missing = sorted(requested - {row["slide"] for row in selected})
        if missing:
            raise RuntimeError(f"Requested slides not in registry: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_path = load_model(config, device)
    manifest_path = output_root / "processing_manifest.json"
    prior_rows: dict[str, dict] = {}
    if manifest_path.is_file() and not args.force:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_rows = {row["slide"]: row for row in prior.get("slides", [])}

    results: list[dict] = []
    for index, row in enumerate(selected, start=1):
        prior = prior_rows.get(row["slide"])
        if prior and prior.get("status") == "complete":
            print(f"Skipping completed slide {row['slide']}")
            results.append(prior)
            continue
        print(f"Processing slide {index}/{len(selected)}: {row['slide']}")
        result = process_slide(row, config, model, device, embedding_root)
        results.append(result)
        manifest = {
            "schema_version": "1.0",
            "dataset": config["dataset"],
            "scientific_scope": config["scientific_scope"],
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "device": str(device),
            "slides": results,
            "completed_count": len(results),
            "requested_count": len(selected),
            "passed": len(results) == len(selected),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_csv(output_root / "processing_manifest.csv", results)

    print(json.dumps({
        "registry_count": len(registry),
        "processed_count": len(results),
        "passed": len(results) == len(selected),
        "test_boundary_status": "UNTOUCHED_UNTIL_FINAL_EVALUATION",
    }, indent=2))


if __name__ == "__main__":
    main()
