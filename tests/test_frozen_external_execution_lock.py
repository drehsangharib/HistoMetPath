import yaml

def test_external_execution_lock_is_strict():
 cfg=yaml.safe_load(open('configs/evaluation/frozen_external_execution_lock.yaml',encoding='utf-8'))
 assert cfg['execution_count_limit']==1
 assert cfg['prohibit_model_refitting'] is True
 assert cfg['prohibit_threshold_refitting'] is True
 assert cfg['prohibit_sampler_changes'] is True
 assert len(cfg['selection_limitations'])>=2
