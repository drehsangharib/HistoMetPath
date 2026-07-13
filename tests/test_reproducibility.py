from experiments.experiment_manifest import build_manifest

def test_manifest_generation():
    m = build_manifest('repro', {'seed': 42}, [])
    assert m['config']['seed'] == 42
