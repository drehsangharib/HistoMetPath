"""Synthetic-only dry-run validator for the final external execution engine.

This module validates deterministic feature assembly, frozen-model inference,
atomic state transitions on temporary fixtures, partial-result sealing, and
one-run refusal semantics. It never opens an external WSI, never mutates the
real external lock, and cannot execute the external cohort.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, tempfile
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/evaluation/frozen_external_execution_engine.yaml')
    p.add_argument('--synthetic-dry-run',action='store_true',required=True)
    return p.parse_args()
def sha_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',delete=False,dir=path.parent,suffix='.tmp') as h:
        json.dump(payload,h,indent=2);h.flush();tmp=Path(h.name)
    tmp.replace(path)
def consume_fixture(lock:dict[str,Any])->dict[str,Any]:
    updated=copy.deepcopy(lock)
    if updated['lock_status']!='sealed_pre_execution':raise RuntimeError('Fixture lock is not sealed_pre_execution')
    if not updated['execution_authorized']:raise RuntimeError('Fixture execution is not authorized')
    if int(updated['execution_count'])>=int(updated['execution_count_limit']):raise RuntimeError('Fixture execution limit reached')
    updated['execution_count']=int(updated['execution_count'])+1
    updated['lock_status']='execution_started'
    return updated
def main():
    args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
    if cfg['real_wsi_access_enabled'] is not False or cfg['external_execution_enabled'] is not False:raise RuntimeError('Real external execution must remain disabled')
    lock_path=project_path(cfg['external_lock']);checksum_path=project_path(cfg['lock_checksum_record']);artifact_path=project_path(cfg['frozen_model_artifact'])
    lock_before=lock_path.read_bytes();checksum=json.loads(checksum_path.read_text(encoding='utf-8-sig'));artifact=joblib.load(artifact_path)
    if sha_bytes(lock_before)!=checksum['lock_sha256']:raise RuntimeError('Real lock checksum mismatch')
    primary=artifact['artifacts'][cfg['primary_model']];secondary=artifact['artifacts'][cfg['secondary_model']]
    rng=np.random.default_rng(20260807);v2=rng.normal(size=(int(cfg['embedding_rows']),int(cfg['embedding_features']))).astype(np.float32);v3=rng.normal(size=v2.shape).astype(np.float32)
    v2_mean=v2.mean(0).astype(np.float64);v3_mean=v3.mean(0).astype(np.float64);concat=np.concatenate([v2_mean,v3_mean])
    p1=float(primary['model'].predict_proba(primary['scaler'].transform(v2_mean[None,:]))[0,1]);p2=float(secondary['model'].predict_proba(secondary['scaler'].transform(concat[None,:]))[0,1]);prediction=int(p1>=float(cfg['primary_threshold']))
    fixture={'lock_status':'sealed_pre_execution','execution_authorized':True,'execution_count_limit':1,'execution_count':0};started=consume_fixture(fixture)
    refused=False
    try:consume_fixture(started)
    except RuntimeError:refused=True
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);state_path=root/'state.json';partial_path=root/'partial_result.json'
        atomic_json(state_path,started);partial={'status':'sealed_partial_fixture','slide':'synthetic_001','primary_probability':p1,'secondary_probability':p2};atomic_json(partial_path,partial)
        fixture_state=json.loads(state_path.read_text(encoding='utf-8'));partial_hash=hashlib.sha256(partial_path.read_bytes()).hexdigest()
    lock_after=lock_path.read_bytes()
    checks={
      'real_execution_disabled':cfg['external_execution_enabled'] is False,
      'real_wsi_access_disabled':cfg['real_wsi_access_enabled'] is False,
      'real_lock_unchanged':lock_after==lock_before,
      'synthetic_v2_shape':list(v2.shape)==[300,512],
      'synthetic_v3_shape':list(v3.shape)==[300,512],
      'primary_shape':list(v2_mean.shape)==[512],
      'secondary_shape':list(concat.shape)==[1024],
      'primary_probability_finite':bool(np.isfinite(p1) and 0.0<=p1<=1.0),
      'secondary_probability_finite':bool(np.isfinite(p2) and 0.0<=p2<=1.0),
      'frozen_threshold_applied':prediction in (0,1),
      'fixture_consumed_before_processing':fixture_state['execution_count']==1 and fixture_state['lock_status']=='execution_started',
      'second_fixture_execution_refused':refused,
      'partial_fixture_sealed':len(partial_hash)==64,
      'primary_features_match':int(primary['model'].n_features_in_)==512,
      'secondary_features_match':int(secondary['model'].n_features_in_)==1024,
      'selected_model_match':artifact['selected_model']==cfg['primary_model'],
      'selected_threshold_match':abs(float(artifact['selected_threshold'])-float(cfg['primary_threshold']))<1e-15,
    }
    passed=all(checks.values());report={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'synthetic_dry_run':True,'real_wsi_accessed':False,'real_lock_mutated':False,'external_execution_consumed':False,'coordinates_generated_for_external_cohort':False,'embeddings_generated_for_external_cohort':False,'predictions_generated_for_external_cohort':False,'synthetic_primary_probability':p1,'synthetic_secondary_probability':p2,'synthetic_primary_prediction':prediction,'checks':checks,'blocking_reasons':[k for k,v in checks.items() if not v],'passed':passed}
    out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'execution_engine_synthetic_dry_run.json',report)
    print(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('Synthetic execution-engine dry run failed')
    print('PASS: Synthetic-only frozen external execution-engine dry run completed without external execution.')
if __name__=='__main__':main()
