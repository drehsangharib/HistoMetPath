"""Final one-time frozen external execution engine, disabled by default.

The real execution path is fully implemented but requires both explicit config
enablement and the exact execution token. The committed milestone exercises only
preflight and synthetic integration modes; neither mode opens an external WSI,
mutates the real lock, or consumes the authorized execution.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler_v2 import sample_slide as sample_v2
from core.wsi.run_camelyon16_spatial_sampler_v3 import sample_slide as sample_v3
from core.wsi.materialize_camelyon16_dual_view_embeddings import build_encoder, embed

def utc_now()->str:return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def atomic_json(path:Path,payload:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',delete=False,dir=path.parent,suffix='.tmp') as h:
  json.dump(payload,h,indent=2);h.flush();os.fsync(h.fileno());tmp=Path(h.name)
 tmp.replace(path)
def consume_lock(lock:dict[str,Any])->dict[str,Any]:
 updated=copy.deepcopy(lock)
 if updated.get('lock_status')!='sealed_pre_execution':raise RuntimeError('External lock is not sealed_pre_execution')
 if updated.get('execution_authorized') is not True:raise RuntimeError('External execution is not authorized')
 if int(updated.get('execution_count',-1))>=int(updated.get('execution_count_limit',-1)):raise RuntimeError('External execution limit has been reached')
 updated['execution_count']=int(updated['execution_count'])+1;updated['lock_status']='execution_started';updated['execution_started_utc']=utc_now();return updated
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation/frozen_external_execution_engine_runtime.yaml');m=p.add_mutually_exclusive_group(required=True);m.add_argument('--preflight-only',action='store_true');m.add_argument('--synthetic-integration-test',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--execution-token',default='');return p.parse_args()
def load_contract(cfg):
 keys=['external_lock','lock_checksum_record','external_processing_manifest','runner_preflight','activation_preflight','synthetic_dry_run','frozen_model_artifact','embedding_config','primary_sampler_config','secondary_sampler_config'];paths={k:project_path(cfg[k]) for k in keys};missing=[f'{k}: {v}' for k,v in paths.items() if not v.is_file()]
 if missing:raise FileNotFoundError('Missing final-engine evidence:\n'+'\n'.join(missing))
 lock_bytes=paths['external_lock'].read_bytes();lock=json.loads(lock_bytes.decode('utf-8'));checksum=json.loads(paths['lock_checksum_record'].read_text(encoding='utf-8-sig'));processing=json.loads(paths['external_processing_manifest'].read_text(encoding='utf-8'));runner=json.loads(paths['runner_preflight'].read_text(encoding='utf-8'));activation=json.loads(paths['activation_preflight'].read_text(encoding='utf-8'));dry=json.loads(paths['synthetic_dry_run'].read_text(encoding='utf-8'));artifact=joblib.load(paths['frozen_model_artifact']);emb_cfg=yaml.safe_load(paths['embedding_config'].read_text(encoding='utf-8-sig'));v2_cfg=yaml.safe_load(paths['primary_sampler_config'].read_text(encoding='utf-8-sig'));v3_cfg=yaml.safe_load(paths['secondary_sampler_config'].read_text(encoding='utf-8-sig'))
 return paths,lock_bytes,lock,checksum,processing,runner,activation,dry,artifact,emb_cfg,v2_cfg,v3_cfg
def contract_checks(cfg,paths,lock_bytes,lock,checksum,processing,runner,activation,dry,artifact,emb_cfg,v2_cfg,v3_cfg):
 primary=artifact['artifacts'][cfg['primary_model']];secondary=artifact['artifacts'][cfg['secondary_model']]
 return {'lock_checksum':hashlib.sha256(lock_bytes).hexdigest()==checksum.get('lock_sha256'),'lock_sealed':lock.get('lock_status')=='sealed_pre_execution','authorized':lock.get('execution_authorized') is True,'count_zero':int(lock.get('execution_count',-1))==0,'limit_one':int(lock.get('execution_count_limit',-1))==1,'slides_twenty':len(processing.get('slides',[]))==int(cfg['expected_slides']),'runner_ready':runner.get('runner_ready_for_one_time_execution') is True,'activation_ready':activation.get('activation_contract_ready') is True,'activation_disabled':activation.get('activation_execution_enabled') is False,'dry_run_passed':dry.get('passed') is True,'dry_run_nonconsequential':dry.get('real_wsi_accessed') is False and dry.get('real_lock_mutated') is False and dry.get('external_execution_consumed') is False,'v2_budget':int(v2_cfg['max_tiles_per_slide'])==300,'v3_budget':int(v3_cfg['max_tiles_per_slide'])==300,'embedding_dimension':int(emb_cfg['embedding_dimension'])==512,'primary_features':int(primary['model'].n_features_in_)==512,'secondary_features':int(secondary['model'].n_features_in_)==1024,'selected_model':artifact['selected_model']==cfg['primary_model'],'selected_threshold':abs(float(artifact['selected_threshold'])-float(cfg['primary_threshold']))<1e-15}
def validate_coordinate_set(result:dict[str,Any],expected:int)->np.ndarray:
 coords=np.asarray([[t['x'],t['y']] for t in result['tiles']],dtype=np.int64)
 if coords.shape!=(expected,2):raise RuntimeError(f"{result['slide']}: coordinate shape {coords.shape}")
 return coords
def predict_slide(row,cfg,v2_cfg,v3_cfg,emb_cfg,encoder,device,artifact):
 v2_result=sample_v2(row,v2_cfg);v3_result=sample_v3(row,v3_cfg);expected=int(cfg['expected_coordinates_per_view']);v2_coords=validate_coordinate_set(v2_result,expected);v3_coords=validate_coordinate_set(v3_result,expected);v2_emb=embed(encoder,row,v2_coords,emb_cfg,device);v3_emb=embed(encoder,row,v3_coords,emb_cfg,device)
 shape=tuple(cfg['expected_embedding_shape'])
 if v2_emb.shape!=shape or v3_emb.shape!=shape:raise RuntimeError(f"{row['slide']}: embedding contract failed")
 v2_mean=v2_emb.mean(0).astype(np.float64);v3_mean=v3_emb.mean(0).astype(np.float64);concat=np.concatenate([v2_mean,v3_mean]);primary=artifact['artifacts'][cfg['primary_model']];secondary=artifact['artifacts'][cfg['secondary_model']];p1=float(primary['model'].predict_proba(primary['scaler'].transform(v2_mean[None,:]))[0,1]);p2=float(secondary['model'].predict_proba(secondary['scaler'].transform(concat[None,:]))[0,1])
 return {'slide':row['slide'],'label':row['label'],'primary_probability':p1,'secondary_probability':p2,'primary_prediction':int(p1>=float(cfg['primary_threshold'])),'v2_coordinates':v2_coords,'v3_coordinates':v3_coords,'v2_embeddings':v2_emb,'v3_embeddings':v3_emb}
def synthetic_integration(cfg,lock):
 started=consume_lock({'lock_status':'sealed_pre_execution','execution_authorized':True,'execution_count_limit':1,'execution_count':0});refused=False
 try:consume_lock(started)
 except RuntimeError:refused=True
 rng=np.random.default_rng(20260808);v2=rng.normal(size=(300,512)).astype(np.float32);v3=rng.normal(size=(300,512)).astype(np.float32)
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);atomic_json(root/'execution_state.json',started);atomic_json(root/'partial.json',{'status':'sealed_partial','slide':'synthetic','sha256':hashlib.sha256(v2.tobytes()+v3.tobytes()).hexdigest()});state=json.loads((root/'execution_state.json').read_text(encoding='utf-8'));partial=json.loads((root/'partial.json').read_text(encoding='utf-8'))
 checks={'fixture_count_one':state['execution_count']==1,'fixture_started':state['lock_status']=='execution_started','second_refused':refused,'v2_shape':v2.shape==(300,512),'v3_shape':v3.shape==(300,512),'partial_sealed':len(partial['sha256'])==64,'real_lock_still_zero':int(lock['execution_count'])==0}
 return {'synthetic_integration_test':True,'checks':checks,'passed':all(checks.values()),'real_wsi_accessed':False,'real_lock_mutated':False,'external_execution_consumed':False}
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));data=load_contract(cfg);paths,lock_bytes,lock,checksum,processing,runner,activation,dry,artifact,emb_cfg,v2_cfg,v3_cfg=data;checks=contract_checks(cfg,*data);ready=all(checks.values());out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
 if args.preflight_only:
  report={'mode':'preflight_only','engine_ready':ready,'external_execution_enabled':cfg['external_execution_enabled'],'real_wsi_access_enabled':cfg['real_wsi_access_enabled'],'execution_count_consumed':False,'inference_executed':False,'checks':checks,'blocking_reasons':[k for k,v in checks.items() if not v],'passed':ready};atomic_json(out/'final_engine_preflight.json',report);print(json.dumps(report,indent=2));return
 if args.synthetic_integration_test:
  report=synthetic_integration(cfg,lock);atomic_json(out/'final_engine_synthetic_integration.json',report);print(json.dumps(report,indent=2));return
 if not ready:raise RuntimeError('Final engine contract is not ready')
 if cfg['external_execution_enabled'] is not True or cfg['real_wsi_access_enabled'] is not True:raise RuntimeError('Final external execution remains disabled; execution was not consumed')
 if args.execution_token!=cfg['required_execution_token']:raise RuntimeError('Invalid execution token; execution was not consumed')
 started=consume_lock(lock);atomic_json(paths['external_lock'],started);atomic_json(paths['lock_checksum_record'],{'lock_sha256':sha(paths['external_lock']),'execution_count':1,'lock_status':'execution_started','recorded_utc':utc_now()})
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');encoder,_=build_encoder(emb_cfg,device);results=[]
 try:
  for row in processing['slides']:
   item=predict_slide(row,cfg,v2_cfg,v3_cfg,emb_cfg,encoder,device,artifact)
   slide_root=out/'slides'/row['slide'];slide_root.mkdir(parents=True,exist_ok=True);np.save(slide_root/'v2_coordinates.npy',item.pop('v2_coordinates'),allow_pickle=False);np.save(slide_root/'v3_coordinates.npy',item.pop('v3_coordinates'),allow_pickle=False);np.save(slide_root/'v2_embeddings.npy',item.pop('v2_embeddings'),allow_pickle=False);np.save(slide_root/'v3_embeddings.npy',item.pop('v3_embeddings'),allow_pickle=False);results.append(item);atomic_json(out/'partial_results.json',{'status':'execution_started','completed_count':len(results),'results':results})
  finished=copy.deepcopy(started);finished['lock_status']='execution_completed';finished['execution_completed_utc']=utc_now();atomic_json(paths['external_lock'],finished);atomic_json(out/'external_predictions.json',{'status':'execution_completed','results':results,'primary_threshold':cfg['primary_threshold']})
 except Exception as exc:
  failed=copy.deepcopy(started);failed['lock_status']='execution_failed_sealed';failed['failure_utc']=utc_now();failed['failure_type']=type(exc).__name__;failed['failure_message']=str(exc);atomic_json(paths['external_lock'],failed);atomic_json(out/'execution_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'completed_results':results});raise
if __name__=='__main__':main()
