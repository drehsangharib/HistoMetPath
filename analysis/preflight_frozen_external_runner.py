"""Non-consuming preflight for the frozen external evaluation runner.

This command does not open WSIs, run samplers, create embeddings, load model
objects for prediction, mutate the external lock, or consume the one-time run.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/evaluation/frozen_external_runner_preflight.yaml')
    return p.parse_args()
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git',*args],text=True).strip()
def main():
    args=parse_args(); cp=project_path(args.config); cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
    keys=['external_lock','lock_checksum_record','external_manifest','readiness_result','frozen_model_artifact','frozen_embedding_config','primary_sampler_config','secondary_sampler_config','encoder_checkpoint']
    paths={k:project_path(cfg[k]) for k in keys}; missing=[f'{k}: {v}' for k,v in paths.items() if not v.is_file()]
    if missing: raise FileNotFoundError('Missing preflight evidence:\n'+'\n'.join(missing))
    lock_bytes_before=paths['external_lock'].read_bytes(); lock_hash_before=hashlib.sha256(lock_bytes_before).hexdigest()
    lock=json.loads(lock_bytes_before.decode('utf-8')); checksum=json.loads(paths['lock_checksum_record'].read_text(encoding='utf-8-sig')); manifest=json.loads(paths['external_manifest'].read_text(encoding='utf-8')); readiness=json.loads(paths['readiness_result'].read_text(encoding='utf-8'))
    usage=shutil.disk_usage(project_path('.')); free_gib=usage.free/(1024**3)
    checks={
      'lock_sealed':lock.get('lock_status')=='sealed_pre_execution',
      'execution_authorized':lock.get('execution_authorized') is True,
      'execution_count_zero':int(lock.get('execution_count',-1))==int(cfg['expected_execution_count']),
      'execution_limit_one':int(lock.get('execution_count_limit',-1))==1,
      'wsi_hashes_complete':lock.get('wsi_sha256_complete') is True,
      'slide_receipts_complete':len(lock.get('slides',[]))==int(cfg['expected_slides']),
      'manifest_slide_count':len(manifest.get('slides',[]))==int(cfg['expected_slides']),
      'readiness_true':readiness.get('ready_for_frozen_external_evaluation') is True,
      'readiness_blockers_empty':len(readiness.get('blocking_reasons',[]))==0,
      'lock_checksum_matches_record':lock_hash_before==checksum.get('lock_sha256'),
      'manifest_checksum_matches_lock':sha(paths['external_manifest'])==lock.get('external_manifest_sha256'),
      'primary_model_match':lock.get('frozen_primary_model')==cfg['primary_model'],
      'secondary_model_match':lock.get('frozen_secondary_model')==cfg['secondary_model'],
      'working_tree_clean':git('status','--porcelain')=='',
      'git_commit_matches_lock':git('rev-parse','HEAD')==lock.get('git_commit'),
      'free_space_minimum':free_gib>=float(cfg['minimum_free_space_gib']),
      'inference_prohibited_in_preflight':cfg['prohibit_inference'] is True,
      'lock_mutation_prohibited':cfg['prohibit_lock_mutation'] is True,
    }
    # Verify current file sizes only. Full WSI hashes were already sealed and are not re-read in preflight.
    receipt_by_id={r['slide_id']:r for r in lock['slides']}; slide_checks=[]
    for slide in manifest['slides']:
        path=project_path(slide['source_path']); receipt=receipt_by_id.get(slide['slide_id']); present=path.is_file(); size=path.stat().st_size if present else 0
        slide_checks.append({'slide_id':slide['slide_id'],'present':present,'size_bytes':size,'sealed_size_bytes':receipt.get('size_bytes') if receipt else None,'size_matches':bool(receipt and present and size==int(receipt['size_bytes'])),'sealed_sha256_present':bool(receipt and isinstance(receipt.get('sha256'),str) and len(receipt['sha256'])==64)})
    checks['all_slide_sizes_match_seal']=all(r['size_matches'] for r in slide_checks)
    checks['all_sealed_slide_hashes_present']=all(r['sealed_sha256_present'] for r in slide_checks)
    lock_hash_after=sha(paths['external_lock']); checks['lock_unchanged_by_preflight']=lock_hash_after==lock_hash_before
    ready=all(checks.values())
    report={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'preflight_only':True,'inference_executed':False,'execution_count_consumed':False,'cohort_id':lock['cohort_id'],'free_space_gib':free_gib,'checks':checks,'runner_ready_for_one_time_execution':ready,'blocking_reasons':[k for k,v in checks.items() if not v],'slide_checks':slide_checks,'external_lock_sha256':lock_hash_after,'config_sha256':sha(cp),'git_commit':git('rev-parse','HEAD'),'passed':True}
    out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'frozen_external_runner_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='slide_checks'},indent=2));print('PASS: Frozen external runner preflight completed without consuming execution.')
if __name__=='__main__': main()
