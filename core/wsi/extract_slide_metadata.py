from pathlib import Path
import json

from core.wsi.openslide_utils import extract_metadata


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "camelyon16"

VALID_SUFFIXES = {
    ".tif",
    ".tiff",
}


def main():
    slides = []

    if DATA_ROOT.exists():
        for file in DATA_ROOT.rglob("*"):
            if (
                file.is_file()
                and file.suffix.lower()
                in VALID_SUFFIXES
            ):
                slides.append(
                    extract_metadata(file)
                )

    output = {
        "dataset": "CAMELYON16",
        "slide_count": len(slides),
        "slides": slides,
    }

    output_dir = (
        ROOT
        / "outputs"
        / "camelyon16"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "slide_metadata.json"
    )

    output_file.write_text(
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
