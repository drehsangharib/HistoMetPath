from experiments.experiment_manifest import build_manifest

def test_ok():
    m = build_manifest('exp', {'seed': 42}, [])
    assert m['experiment_name'] == 'exp'
    assert m['config']['seed'] == 42
