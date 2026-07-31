"""Create a pre-execution lock for a one-time frozen external pilot evaluation."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation/frozen_external_execution_lock.yaml');p.add_argument('--quick',action='store_true',help='Record file sizes without full WSI SHA-256; not execution-ready.');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));keys=['external_manifest','readiness_result','development_decision','final_test_lock','frozen_model_artifact','frozen_embedding_config','frozen_primary_sampler_config','frozen_secondary_sampler_config','frozen_encoder_checkpoint'];paths={k:project_path(cfg[k]) for k in keys}
 missing=[f'{k}: {v}' for k,v in paths.items() if not v.is_file()]
 if missing:raise FileNotFoundError('Missing frozen evidence:\n'+'\n'.join(missing))
 manifest=json.loads(paths['external_manifest'].read_text(encoding='utf-8'));readiness=json.loads(paths['readiness_result'].read_text(encoding='utf-8'));decision=json.loads(paths['development_decision'].read_text(encoding='utf-8'));final_lock=json.loads(paths['final_test_lock'].read_text(encoding='utf-8'))
 slides=manifest['slides'];labels=[s['label'] for s in slides];patients=[s['patient_id'] for s in slides]
 checks={'readiness_true':readiness.get('ready_for_frozen_external_evaluation') is True,'readiness_blockers_empty':len(readiness.get('blocking_reasons',[]))==0,'manifest_hash_matches_readiness':sha(paths['external_manifest'])==readiness.get('candidate_manifest_sha256'),'development_complete':decision.get('status')=='internal_development_cycle_complete' and decision.get('passed') is True,'final_test_immutable':decision['frozen_decisions']['completed_final_test_is_immutable'] is True and final_lock.get('executed_once') is True,'slide_count':len(slides)==int(cfg['expected_slides']),'normal_count':labels.count('normal')==int(cfg['expected_normal_slides']),'tumor_count':labels.count('tumor')==int(cfg['expected_tumor_slides']),'unique_patients':len(set(patients))==int(cfg['expected_unique_patients']),'primary_model_match':decision['frozen_decisions']['primary_development_baseline']==cfg['primary_model'],'secondary_model_match':decision['frozen_decisions']['secondary_exploratory_candidate']==cfg['secondary_model']}
 if not all(checks.values()):raise RuntimeError(f'Pre-execution boundary failed: {checks}')
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);lock_path=out/'FROZEN_EXTERNAL_EVALUATION.lock'
 if lock_path.exists():raise RuntimeError(f'Lock already exists and will not be overwritten: {lock_path}')
 slide_receipts=[]
 for index,slide in enumerate(slides,1):
  path=project_path(slide['source_path'])
  if not path.is_file():raise FileNotFoundError(path)
  print(f'[{index}/{len(slides)}] sealing {slide["slide_id"]}: {path.name}',flush=True)
  slide_receipts.append({'slide_id':slide['slide_id'],'patient_id':slide['patient_id'],'label':slide['label'],'acquisition_site':slide['acquisition_site'],'source_path':slide['source_path'],'size_bytes':path.stat().st_size,'sha256':None if args.quick else sha(path)})
 full_hashes=not args.quick
 lock={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'lock_status':'sealed_pre_execution' if full_hashes else 'draft_quick_inventory','execution_authorized':full_hashes,'execution_count_limit':int(cfg['execution_count_limit']),'execution_count':0,'cohort_id':manifest['cohort_id'],'selection_design':cfg['selection_design'],'selection_limitations':cfg['selection_limitations'],'frozen_primary_model':cfg['primary_model'],'frozen_secondary_model':cfg['secondary_model'],'prohibitions':{k:v for k,v in cfg.items() if k.startswith('prohibit_')},'boundary_checks':checks,'git_commit':git('rev-parse','HEAD'),'git_branch':git('rev-parse','--abbrev-ref','HEAD'),'working_tree_clean':git('status','--porcelain')=='','external_manifest_sha256':sha(paths['external_manifest']),'readiness_result_sha256':sha(paths['readiness_result']),'development_decision_sha256':sha(paths['development_decision']),'final_test_lock_sha256':sha(paths['final_test_lock']),'frozen_artifact_sha256':{k:sha(v) for k,v in paths.items() if k not in {'external_manifest','readiness_result','development_decision','final_test_lock'}},'slides':slide_receipts,'wsi_sha256_complete':full_hashes,'passed':True}
 lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8');(out/'external_slide_receipts.json').write_text(json.dumps(slide_receipts,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in lock.items() if k!='slides'},indent=2));print('PASS: Frozen external pre-execution lock created.' if full_hashes else 'PASS: Quick external inventory created; execution remains unauthorized.')
if __name__=='__main__':main()
