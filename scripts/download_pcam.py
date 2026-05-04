import urllib.request
from pathlib import Path

ZENODO_RECORD = "2546921"
BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"

FILES = [
    "camelyonpatch_level_2_split_train_x.h5.gz",
    "camelyonpatch_level_2_split_train_y.h5.gz",
    "camelyonpatch_level_2_split_valid_x.h5.gz",
    "camelyonpatch_level_2_split_valid_y.h5.gz",
    "camelyonpatch_level_2_split_test_x.h5.gz",
    "camelyonpatch_level_2_split_test_y.h5.gz",
]

def download_pcam(destination="data/pcam"):
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        url = f"{BASE_URL}/{fname}"
        out = dst / fname

        if out.exists():
            print(f"[✓] {fname} already exists, skipping")
            continue

        print(f"[↓] Downloading {fname}")
        urllib.request.urlretrieve(url, out)

    print("\n✅ PatchCamelyon download complete")

if __name__ == "__main__":
    download_pcam()
