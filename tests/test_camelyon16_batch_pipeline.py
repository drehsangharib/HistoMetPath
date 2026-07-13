from pathlib import Path

from core.wsi.run_camelyon16_batch_pipeline import select_level, stratified_split


def test_select_level_by_physical_resolution():
    level, downsample, effective_mpp = select_level(
        0.243094,
        0.243094,
        [1.0, 2.0, 4.0, 8.0],
        0.5,
    )
    assert level == 1
    assert downsample == 2.0
    assert abs(effective_mpp - 0.486188) < 1e-6


def test_deterministic_stratified_split():
    slides = []
    for label in ("normal", "tumor"):
        for index in range(11):
            slides.append(
                {
                    "slide": f"{label}_{index:03d}",
                    "label": label,
                    "path": str(Path("data") / f"{label}_{index:03d}.tif"),
                    "size_bytes": index + 1,
                }
            )
    counts = {"train": 7, "validation": 2, "test": 2}
    first = stratified_split(slides, counts, seed=42)
    second = stratified_split(slides, counts, seed=42)
    assert first == second
    for split_name, expected_count in counts.items():
        for label in ("normal", "tumor"):
            observed = sum(
                row["split"] == split_name and row["label"] == label
                for row in first
            )
            assert observed == expected_count
