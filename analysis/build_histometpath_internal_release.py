"""Build a source-only HistoMetPath internal development release archive."""
from __future__ import annotations
import argparse,fnmatch,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/release/histometpath_internal_release.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def excluded(path,patterns):
 text=path.as_posix();return any(fnmatch.fnmatch(path.name,p) or fnmatch.fnmatch(text,p) or p in path.parts for p in patterns)
def git(command):
 return subprocess.check_output(['git',*command],text=True).strip()
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));decision_path=project_path(cfg['decision_result']);report_path=project_path(cfg['decision_report']);lock_path=project_path(cfg['final_test_lock'])
 for path in [decision_path,report_path,lock_path]:
  if not path.is_file():raise FileNotFoundError(path)
 decision=json.loads(decision_path.read_text(encoding='utf-8'));lock=json.loads(lock_path.read_text(encoding='utf-8'))
 if decision['status']!='internal_development_cycle_complete' or decision['passed'] is not True:raise RuntimeError('Development decision is not frozen')
 if lock.get('executed_once') is not True:raise RuntimeError('Final-test lock invalid')
 out=project_path(cfg['output_root']);stage=out/'stage';archive=out/'HistoMetPath_internal_development_release.zip'
 if stage.exists():shutil.rmtree(stage)
 stage.mkdir(parents=True);patterns=list(cfg['exclude_patterns']);copied=[]
 for item in cfg['include_repository_files']:
  source=project_path(item)
  if not source.exists():continue
  if source.is_file():
   if not excluded(source,patterns):dest=stage/source.name;shutil.copy2(source,dest);copied.append(dest.relative_to(stage).as_posix())
  else:
   for path in source.rglob('*'):
    if not path.is_file() or excluded(path.relative_to(project_path('.')),patterns):continue
    relative=path.relative_to(project_path('.'));dest=stage/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);copied.append(relative.as_posix())
 release_metadata={'schema_version':'1.0','project':cfg['project'],'release_name':cfg['release_name'],'release_status':cfg['release_status'],'git_commit':git(['rev-parse','HEAD']),'git_branch':git(['rev-parse','--abbrev-ref','HEAD']),'working_tree_clean':git(['status','--porcelain'])=='','primary_development_baseline':decision['frozen_decisions']['primary_development_baseline'],'secondary_exploratory_candidate':decision['frozen_decisions']['secondary_exploratory_candidate'],'final_test_immutable':True,'new_performance_claim_requires_untouched_cohort':True,'decision_result_sha256':sha(decision_path),'decision_report_sha256':sha(report_path),'final_test_lock_sha256':sha(lock_path),'source_file_count':len(copied),'passed':True}
 (stage/'RELEASE_METADATA.json').write_text(json.dumps(release_metadata,indent=2),encoding='utf-8');shutil.copy2(decision_path,stage/'DEVELOPMENT_DECISION.json');shutil.copy2(report_path,stage/'DEVELOPMENT_DECISION_REPORT.md')
 manifest=[]
 for path in sorted(stage.rglob('*')):
  if path.is_file():manifest.append({'path':path.relative_to(stage).as_posix(),'size_bytes':path.stat().st_size,'sha256':sha(path)})
 (stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 if archive.exists():archive.unlink()
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in sorted(stage.rglob('*')):
   if path.is_file():z.write(path,path.relative_to(stage).as_posix())
 with zipfile.ZipFile(archive,'r') as z:bad=z.testzip()
 if bad:raise RuntimeError(f'Corrupt archive member: {bad}')
 result={**release_metadata,'archive':str(archive),'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,'manifest_records':len(manifest)};(out/'internal_release_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2));print('PASS: HistoMetPath internal development release built.')
if __name__=='__main__':main()
