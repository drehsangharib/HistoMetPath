from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "camelyon16"

VALID_SUFFIXES = {
    ".tif",
    ".tiff",
}


def discover_slides(root):
    rows = []

    if not root.exists():
        return rows

    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in VALID_SUFFIXES:
            rows.append(
                {
                    "slide_name": file.name,
                    "path": str(file),
                }
            )

    return rows


def main():
    slides = discover_slides(DATA_ROOT)

    output = {
        "dataset": "CAMELYON16",
        "slide_count": len(slides),
        "slides": slides,
    }

    out_dir = ROOT / "outputs" / "camelyon16"

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = out_dir / "slide_manifest.json"

    out_file.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
