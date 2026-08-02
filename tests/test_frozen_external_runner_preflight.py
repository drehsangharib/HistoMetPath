import yaml

def test_preflight_cannot_run_inference_or_mutate_lock():
    cfg=yaml.safe_load(open('configs/evaluation/frozen_external_runner_preflight.yaml',encoding='utf-8'))
    assert cfg['prohibit_inference'] is True
    assert cfg['prohibit_lock_mutation'] is True
    assert cfg['expected_execution_count']==0
    assert cfg['minimum_free_space_gib']>=15
