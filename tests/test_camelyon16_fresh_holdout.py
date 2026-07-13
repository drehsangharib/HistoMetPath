from core.wsi.run_camelyon16_expanded_batch_pipeline import assign_fixed_splits


def test_fixed_fresh_holdout_assignment():
    slides = []
    for label in ("normal", "tumor"):
        identifiers = list(range(1, 21)) + [100]
        for number in identifiers:
            slides.append(
                {
                    "slide": f"{label}_{number:03d}",
                    "label": label,
                    "path": f"{label}_{number:03d}.tif",
                    "size_bytes": 1,
                }
            )
    config = {
        "fixed_validation_slides": [
            "normal_015", "normal_016", "normal_017",
            "tumor_015", "tumor_016", "tumor_017",
        ],
        "fixed_test_slides": [
            "normal_018", "normal_019", "normal_020",
            "tumor_018", "tumor_019", "tumor_020",
        ],
        "expected_counts_per_class": {
            "train": 15,
            "validation": 3,
            "test": 3,
        },
    }
    assigned = assign_fixed_splits(slides, config)
    assert len(assigned) == 42
    for slide in config["fixed_test_slides"]:
        row = next(item for item in assigned if item["slide"] == slide)
        assert row["split"] == "test"
    for slide in config["fixed_validation_slides"]:
        row = next(item for item in assigned if item["slide"] == slide)
        assert row["split"] == "validation"
