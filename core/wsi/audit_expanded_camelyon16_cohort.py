"""Audit local CAMELYON16 WSI readability with OpenSlide."""

from __future__ import annotations

import json
from pathlib import Path

import openslide


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "camelyon16" / "training"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "camelyon16"
    / "expanded_cohort_openslide_audit.json"
)


def classify_slide(path: Path) -> str:
    if path.name.startswith("normal_"):
        return "normal"
    if path.name.startswith("tumor_"):
        return "tumor"
    return "unknown"


def audit_slide(path: Path) -> dict:
    row = {
        "slide": path.stem,
        "label": classify_slide(path),
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "opened": False,
        "error": None,
    }

    slide = None
    try:
        slide = openslide.OpenSlide(str(path))
        row.update(
            {
                "opened": True,
                "width": int(slide.dimensions[0]),
                "height": int(slide.dimensions[1]),
                "level_count": int(slide.level_count),
                "vendor": slide.properties.get("openslide.vendor"),
                "mpp_x": slide.properties.get("openslide.mpp-x"),
                "mpp_y": slide.properties.get("openslide.mpp-y"),
                "quickhash": slide.properties.get("openslide.quickhash-1"),
            }
        )
    except Exception as error:
        row["error"] = repr(error)
    finally:
        if slide is not None:
            slide.close()

    return row


def main() -> None:
    slides = sorted(path for path in DATA_ROOT.rglob("*.tif") if path.is_file())
    rows = [audit_slide(path) for path in slides]
    failed = [row for row in rows if not row["opened"]]

    output = {
        "schema_version": "1.0",
        "dataset": "CAMELYON16",
        "scientific_scope": "local expanded-cohort WSI readability audit",
        "slide_count": len(rows),
        "normal_count": sum(row["label"] == "normal" for row in rows),
        "tumor_count": sum(row["label"] == "tumor" for row in rows),
        "unknown_count": sum(row["label"] == "unknown" for row in rows),
        "opened_count": sum(bool(row["opened"]) for row in rows),
        "failed_count": len(failed),
        "slides": rows,
        "passed": len(failed) == 0 and len(rows) > 0,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))

    if not rows:
        raise RuntimeError("No CAMELYON16 TIFF slides were found.")
    if failed:
        raise RuntimeError(f"{len(failed)} WSI files could not be opened.")

    print()
    print("PASS: All local CAMELYON16 WSIs opened successfully with OpenSlide.")
    print(f"Audit written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
