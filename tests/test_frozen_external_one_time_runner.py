import yaml

def test_execution_is_disabled_in_scaffold():
    cfg=yaml.safe_load(open('configs/evaluation/frozen_external_one_time_runner.yaml',encoding='utf-8'))
    assert cfg['execute_implementation_enabled'] is False
    assert cfg['execution_count_limit']==1
    assert cfg['required_execution_token']=='I_UNDERSTAND_THIS_CONSUMES_THE_ONLY_EXTERNAL_EXECUTION'
    assert cfg['primary_threshold']==0.2404209436418631

def test_frozen_dimensions():
    cfg=yaml.safe_load(open('configs/evaluation/frozen_external_one_time_runner.yaml',encoding='utf-8'))
    assert cfg['expected_coordinates_per_view']==300
    assert cfg['expected_primary_features']==512
    assert cfg['expected_secondary_features']==1024
