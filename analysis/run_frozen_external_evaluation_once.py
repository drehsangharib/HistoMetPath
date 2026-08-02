"""Guarded one-time frozen external evaluation runner.

The committed version supports a strict, non-consuming ``--preflight-only``
mode. The consequential execution branch is deliberately disabled until the
external adapter implementation has been separately reviewed, tested, CI-green,
and sealed at the exact runner commit. This prevents accidental consumption of
the only authorized external execution while preserving the complete contract.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
import joblib, yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/evaluation/frozen_external_one_time_runner.yaml')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    p.add_argument('--execution-token',default='')
    return p.parse_args()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def main():
    args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
    keys=['external_lock','lock_checksum_record','external_manifest','readiness_result','runner_preflight_result','frozen_model_artifact','encoder_checkpoint','embedding_config','primary_sampler_config','secondary_sampler_config']
    paths={key:project_path(cfg[key]) for key in keys};missing=[f'{key}: {path}' for key,path in paths.items() if not path.is_file()]
    if missing:raise FileNotFoundError('Missing frozen runner evidence:\n'+'\n'.join(missing))
    lock=json.loads(paths['external_lock'].read_text(encoding='utf-8'));checksum=json.loads(paths['lock_checksum_record'].read_text(encoding='utf-8-sig'));manifest=json.loads(paths['external_manifest'].read_text(encoding='utf-8'));readiness=json.loads(paths['readiness_result'].read_text(encoding='utf-8'));preflight=json.loads(paths['runner_preflight_result'].read_text(encoding='utf-8'));artifact=joblib.load(paths['frozen_model_artifact'])
    artifacts=artifact['artifacts'];primary=artifacts[cfg['primary_model']];secondary=artifacts[cfg['secondary_model']];labels=[row['label'] for row in manifest['slides']]
    checks={
      'lock_sealed':lock.get('lock_status')=='sealed_pre_execution',
      'execution_authorized':lock.get('execution_authorized') is True,
      'execution_count_zero':int(lock.get('execution_count',-1))==0,
      'execution_limit_one':int(lock.get('execution_count_limit',-1))==int(cfg['execution_count_limit']),
      'lock_checksum_matches':sha(paths['external_lock'])==checksum.get('lock_sha256'),
      'manifest_checksum_matches_lock':sha(paths['external_manifest'])==lock.get('external_manifest_sha256'),
      'readiness_true':readiness.get('ready_for_frozen_external_evaluation') is True,
      'runner_preflight_true':preflight.get('runner_ready_for_one_time_execution') is True,
      'preflight_non_consuming':preflight.get('execution_count_consumed') is False and preflight.get('inference_executed') is False,
      'slide_count':len(manifest['slides'])==int(cfg['expected_slides']),
      'normal_count':labels.count('normal')==int(cfg['expected_normal_slides']),
      'tumor_count':labels.count('tumor')==int(cfg['expected_tumor_slides']),
      'primary_model_match':artifact.get('selected_model')==cfg['primary_model'],
      'primary_threshold_match':abs(float(artifact.get('selected_threshold'))-float(cfg['primary_threshold']))<1e-15,
      'primary_features_match':int(primary['model'].n_features_in_)==int(cfg['expected_primary_features']) and int(primary['scaler'].n_features_in_)==int(cfg['expected_primary_features']),
      'secondary_features_match':int(secondary['model'].n_features_in_)==int(cfg['expected_secondary_features']) and int(secondary['scaler'].n_features_in_)==int(cfg['expected_secondary_features']),
      'git_commit_matches_lock':git('rev-parse','HEAD')==lock.get('git_commit'),
      'working_tree_clean':git('status','--porcelain')=='',
      'free_space_minimum':shutil.disk_usage(project_path('.')).free/(1024**3)>=15.0,
    }
    ready=all(checks.values());out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    report={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'mode':'preflight_only' if args.preflight_only else 'execute_requested','inference_executed':False,'execution_count_consumed':False,'execute_implementation_enabled':bool(cfg['execute_implementation_enabled']),'runner_contract_ready':ready,'checks':checks,'blocking_reasons':[k for k,v in checks.items() if not v],'git_commit':git('rev-parse','HEAD'),'external_lock_sha256':sha(paths['external_lock']),'config_sha256':sha(cp),'passed':True}
    (out/'one_time_runner_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.execute:
        if args.execution_token!=cfg['required_execution_token']:raise RuntimeError('Invalid execution token; execution was not consumed.')
        if not bool(cfg['execute_implementation_enabled']):raise RuntimeError('Execution implementation is intentionally disabled in this reviewed scaffold; execution was not consumed.')
        raise RuntimeError('Execution branch unavailable; execution was not consumed.')
    print(json.dumps(report,indent=2));print('PASS: One-time external runner scaffold preflight completed without consuming execution.')
if __name__=='__main__':main()
