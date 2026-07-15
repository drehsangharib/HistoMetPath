from pathlib import Path
from analysis.evaluate_camelyon16_fresh_test_once import sha256

def test_sha256_is_stable(tmp_path: Path):
    p=tmp_path/"artifact.bin"; p.write_bytes(b"locked")
    assert sha256(p)==sha256(p)
    assert len(sha256(p))==64
