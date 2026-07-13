from pathlib import Path
import openslide


def open_slide(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return openslide.OpenSlide(str(path))


def extract_metadata(path):
    slide = open_slide(path)

    metadata = {
        "path": str(path),
        "width": slide.dimensions[0],
        "height": slide.dimensions[1],
        "level_count": slide.level_count,
        "level_dimensions": [
            list(dim)
            for dim in slide.level_dimensions
        ],
        "properties": dict(slide.properties),
    }

    slide.close()

    return metadata
