"""Validate a candidate untouched external cohort before any evaluation run."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation/external_cohort_readiness.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def normalize(value):return str(value).strip().lower()
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));existing_path=project_path(cfg['existing_processing_manifest']);lock_path=project_path(cfg['final_test_lock']);decision_path=project_path(cfg['development_decision']);candidate_path=project_path(cfg['candidate_external_manifest'])
 for path in [existing_path,lock_path,decision_path,candidate_path]:
  if not path.is_file():raise FileNotFoundError(path)
 existing=json.loads(existing_path.read_text(encoding='utf-8'));lock=json.loads(lock_path.read_text(encoding='utf-8'));decision=json.loads(decision_path.read_text(encoding='utf-8'));candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
 if lock.get('executed_once') is not True:raise RuntimeError('Final-test receipt invalid')
 if decision.get('status')!='internal_development_cycle_complete' or decision.get('passed') is not True:raise RuntimeError('Frozen decision invalid')
 missing_top=[f for f in cfg['required_candidate_fields'] if f not in candidate];slides=candidate.get('slides',[]);missing_slide=[]
 for index,row in enumerate(slides):
  fields=[f for f in cfg['required_slide_fields'] if f not in row or str(row[f]).strip()=='']
  if fields:missing_slide.append({'index':index,'missing':fields})
 labels=Counter(normalize(r.get('label','')) for r in slides);existing_ids={normalize(r.get('slide','')) for r in existing.get('slides',[])};candidate_ids=[normalize(r.get('slide_id','')) for r in slides];slide_overlap=sorted(set(candidate_ids)&existing_ids);duplicate_slide_ids=sorted([k for k,v in Counter(candidate_ids).items() if k and v>1]);patients=[normalize(r.get('patient_id','')) for r in slides];duplicate_patients=sorted([k for k,v in Counter(patients).items() if k and v>1])
 path_missing=[r.get('source_path','') for r in slides if r.get('source_path') and not project_path(r['source_path']).is_file()]
 checks={'candidate_top_fields_complete':not missing_top,'candidate_slide_fields_complete':not missing_slide,'minimum_total_slides_met':len(slides)>=int(cfg['minimum_total_slides']),'minimum_normal_slides_met':labels['normal']>=int(cfg['minimum_slides_per_class']),'minimum_tumor_slides_met':labels['tumor']>=int(cfg['minimum_slides_per_class']),'labels_allowed':set(labels).issubset(set(cfg['allowed_labels'])),'slide_ids_unique':not duplicate_slide_ids,'patient_ids_unique':(not duplicate_patients) if cfg['require_unique_patient_ids'] else True,'zero_slide_overlap':not slide_overlap,'all_source_files_present':not path_missing,'frozen_primary_matches':decision['frozen_decisions']['primary_development_baseline']==cfg['frozen_primary_model'],'frozen_secondary_matches':decision['frozen_decisions']['secondary_exploratory_candidate']==cfg['frozen_secondary_model'],'final_test_immutable':decision['frozen_decisions']['completed_final_test_is_immutable'] is True}
 ready=all(checks.values());report={'schema_version':'1.0','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'cohort_id':candidate.get('cohort_id'),'total_slides':len(slides),'label_counts':dict(labels),'checks':checks,'ready_for_frozen_external_evaluation':ready,'blocking_reasons':[name for name,value in checks.items() if not value],'missing_top_fields':missing_top,'missing_slide_fields':missing_slide,'duplicate_slide_ids':duplicate_slide_ids,'duplicate_patient_ids':duplicate_patients,'overlapping_slide_ids':slide_overlap,'missing_source_paths':path_missing,'existing_manifest_sha256':sha(existing_path),'candidate_manifest_sha256':sha(candidate_path),'final_test_lock_sha256':sha(lock_path),'development_decision_sha256':sha(decision_path),'config_sha256':sha(cp),'passed':True}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'external_cohort_readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));print('PASS: External-cohort readiness audit completed.')
if __name__=='__main__':main()
