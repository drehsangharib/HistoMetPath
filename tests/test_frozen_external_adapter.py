import yaml

def test_adapter_execution_disabled():
 cfg=yaml.safe_load(open('configs/evaluation/frozen_external_adapter.yaml',encoding='utf-8'))
 assert cfg['adapter_execution_enabled'] is False
 assert cfg['expected_coordinates_per_view']==300
 assert cfg['expected_embedding_shape']==[300,512]
 assert cfg['primary_threshold']==0.2404209436418631
