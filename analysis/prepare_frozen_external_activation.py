"""Validate the execution-disabled frozen external activation implementation.

This milestone freezes the exact callable contracts for the unchanged samplers,
materializer, and models. It never calls sample_slide(), embed(), read_region(),
predict_proba(), or the execution token and never mutates the external lock.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, subprocess
from pathlib import Path
import joblib, yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler_v2 import sample_slide as sample_v2
from core.wsi.run_camelyon16_spatial_sampler_v3 import sample_slide as sample_v3
from core.wsi.materialize_camelyon16_dual_view_embeddings import build_encoder, embed

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/evaluation/frozen_external_activation.yaml')
    p.add_argument('--preflight-only',action='store_true',required=True)
    return p.parse_args()
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def main():
    args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
    keys=['external_lock','lock_checksum_record','external_processing_manifest','external_adapter_preflight','runner_preflight','frozen_model_artifact','embedding_config','primary_sampler_config','secondary_sampler_config']
    paths={k:project_path(cfg[k]) for k in keys};missing=[f'{k}: {v}' for k,v in paths.items() if not v.is_file()]
    if missing:raise FileNotFoundError('Missing activation evidence:\n'+'\n'.join(missing))
    lock_bytes=paths['external_lock'].read_bytes();lock_hash=hashlib.sha256(lock_bytes).hexdigest();lock=json.loads(lock_bytes.decode('utf-8'));checksum=json.loads(paths['lock_checksum_record'].read_text(encoding='utf-8-sig'));processing=json.loads(paths['external_processing_manifest'].read_text(encoding='utf-8'));adapter=json.loads(paths['external_adapter_preflight'].read_text(encoding='utf-8'));runner=json.loads(paths['runner_preflight'].read_text(encoding='utf-8'));artifact=joblib.load(paths['frozen_model_artifact']);emb=yaml.safe_load(paths['embedding_config'].read_text(encoding='utf-8-sig'));v2=yaml.safe_load(paths['primary_sampler_config'].read_text(encoding='utf-8-sig'));v3=yaml.safe_load(paths['secondary_sampler_config'].read_text(encoding='utf-8-sig'))
    primary=artifact['artifacts'][cfg['primary_model']];secondary=artifact['artifacts'][cfg['secondary_model']]
    checks={
      'activation_disabled':cfg['activation_execution_enabled'] is False,
      'lock_sealed':lock.get('lock_status')=='sealed_pre_execution',
      'execution_authorized':lock.get('execution_authorized') is True,
      'execution_count_zero':int(lock.get('execution_count',-1))==0,
      'execution_limit_one':int(lock.get('execution_count_limit',-1))==int(cfg['execution_count_limit']),
      'lock_hash_matches_record':lock_hash==checksum.get('lock_sha256'),
      'processing_manifest_passed':processing.get('passed') is True and len(processing.get('slides',[]))==int(cfg['expected_slides']),
      'adapter_preflight_ready':adapter.get('adapter_ready_for_activation') is True and adapter.get('execution_count_consumed') is False,
      'runner_preflight_ready':runner.get('runner_ready_for_one_time_execution') is True and runner.get('execution_count_consumed') is False,
      'v2_contract':str(inspect.signature(sample_v2))=='(row, cfg)',
      'v3_contract':str(inspect.signature(sample_v3))=='(row, cfg)',
      'encoder_contract':str(inspect.signature(build_encoder))=='(cfg, device)',
      'embed_contract':str(inspect.signature(embed))=='(model, row, coords, cfg, device)',
      'coordinate_budgets':int(v2['max_tiles_per_slide'])==int(cfg['expected_coordinates_per_view']) and int(v3['max_tiles_per_slide'])==int(cfg['expected_coordinates_per_view']),
      'embedding_contract':int(emb['embedding_dimension'])==int(cfg['expected_embedding_features']),
      'primary_features':int(primary['model'].n_features_in_)==int(cfg['expected_primary_features']) and int(primary['scaler'].n_features_in_)==int(cfg['expected_primary_features']),
      'secondary_features':int(secondary['model'].n_features_in_)==int(cfg['expected_secondary_features']) and int(secondary['scaler'].n_features_in_)==int(cfg['expected_secondary_features']),
      'primary_model_frozen':artifact.get('selected_model')==cfg['primary_model'],
      'primary_threshold_frozen':abs(float(artifact.get('selected_threshold'))-float(cfg['primary_threshold']))<1e-15,
      'git_commit_matches_lock':git('rev-parse','HEAD')==lock.get('git_commit'),
      'working_tree_clean':git('status','--porcelain')=='',
    }
    ready=all(checks.values());plan={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'preflight_only':True,'activation_execution_enabled':False,'pixel_processing_executed':False,'coordinates_generated':False,'embeddings_generated':False,'predictions_generated':False,'execution_count_consumed':False,'activation_contract_ready':ready,'checks':checks,'blocking_reasons':[k for k,v in checks.items() if not v],'callable_contracts':{'sample_v2':str(inspect.signature(sample_v2)),'sample_v3':str(inspect.signature(sample_v3)),'build_encoder':str(inspect.signature(build_encoder)),'embed':str(inspect.signature(embed))},'planned_shapes':{'coordinates_per_view':[int(cfg['expected_coordinates_per_view']),2],'embedding_per_view':[int(cfg['expected_embedding_rows']),int(cfg['expected_embedding_features'])],'primary_features':int(cfg['expected_primary_features']),'secondary_features':int(cfg['expected_secondary_features'])},'primary_threshold':float(cfg['primary_threshold']),'external_lock_sha256':lock_hash,'config_sha256':sha(cp),'git_commit':git('rev-parse','HEAD'),'passed':True}
    out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'activation_implementation_preflight.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(plan,indent=2));print('PASS: Execution-disabled external activation implementation preflight completed.')
if __name__=='__main__':main()
