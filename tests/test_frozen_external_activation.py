import inspect,yaml
from core.wsi.run_camelyon16_spatial_sampler_v2 import sample_slide as sample_v2
from core.wsi.run_camelyon16_spatial_sampler_v3 import sample_slide as sample_v3
from core.wsi.materialize_camelyon16_dual_view_embeddings import build_encoder,embed

def load():return yaml.safe_load(open('configs/evaluation/frozen_external_activation.yaml',encoding='utf-8'))
def test_activation_is_disabled_and_dimensions_are_frozen():
 c=load();assert c['activation_execution_enabled'] is False;assert c['execution_count_limit']==1;assert c['expected_coordinates_per_view']==300;assert c['expected_embedding_rows']==300;assert c['expected_embedding_features']==512;assert c['expected_primary_features']==512;assert c['expected_secondary_features']==1024;assert c['primary_threshold']==0.2404209436418631
def test_unchanged_callable_contracts():
 assert str(inspect.signature(sample_v2))=='(row, cfg)';assert str(inspect.signature(sample_v3))=='(row, cfg)';assert str(inspect.signature(build_encoder))=='(cfg, device)';assert str(inspect.signature(embed))=='(model, row, coords, cfg, device)'
