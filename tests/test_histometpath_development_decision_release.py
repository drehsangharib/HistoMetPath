import yaml

def test_frozen_decision_config():
 cfg=yaml.safe_load(open('configs/release/histometpath_development_decision.yaml',encoding='utf-8'))
 decisions=cfg['frozen_decisions']
 assert decisions['primary_development_baseline']=='spatial_v2_mean_pool_lr'
 assert decisions['completed_final_test_is_immutable'] is True
 assert decisions['new_performance_claim_requires_untouched_cohort'] is True
