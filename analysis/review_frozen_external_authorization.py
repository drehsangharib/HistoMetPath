"""Final non-consuming authorization review for the frozen external pilot.

This corrected review validates the final engine structurally with Python AST,
so local variable names and formatting do not affect the outcome. It permits
only its four expected pre-commit files and cannot enable or consume execution.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation/frozen_external_authorization_review.yaml');p.add_argument('--review-only',action='store_true',required=True);return p.parse_args()
def changed_paths()->set[str]:
 result=set()
 for line in subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],text=True).splitlines():
  value=line[3:] if len(line)>=4 else line
  if ' -> ' in value:value=value.split(' -> ',1)[1]
  result.add(value.replace('\\','/'))
 return result
def engine_ast_guards(source:str)->dict[str,bool]:
 tree=ast.parse(source);execute_option=False;consume_call_line=None;encoder_call_line=None;external_flag=False;wsi_flag=False;token_attribute=False;required_token=False
 for node in ast.walk(tree):
  if isinstance(node,ast.Call):
   if isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--execute':execute_option=True
   if isinstance(node.func,ast.Name) and node.func.id=='consume_lock':consume_call_line=getattr(node,'lineno',10**9) if consume_call_line is None else min(consume_call_line,getattr(node,'lineno',10**9))
   if isinstance(node.func,ast.Name) and node.func.id=='build_encoder':encoder_call_line=getattr(node,'lineno',-1) if encoder_call_line is None else min(encoder_call_line,getattr(node,'lineno',-1))
  if isinstance(node,ast.Constant) and node.value=='external_execution_enabled':external_flag=True
  if isinstance(node,ast.Constant) and node.value=='real_wsi_access_enabled':wsi_flag=True
  if isinstance(node,ast.Attribute) and node.attr=='execution_token':token_attribute=True
  if isinstance(node,ast.Constant) and node.value=='required_execution_token':required_token=True
 return {'execute_cli_mode':execute_option,'dual_enablement_guard':external_flag and wsi_flag,'token_guard':token_attribute and required_token,'consume_before_encoder':consume_call_line is not None and encoder_call_line is not None and consume_call_line<encoder_call_line,'second_execution_refusal':'execution limit has been reached' in source,'failure_sealing':'execution_failed_sealed' in source,'completion_sealing':'execution_completed' in source}
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
 keys=['external_lock','lock_checksum_record','runner_preflight','activation_preflight','final_engine_preflight','final_engine_synthetic_integration','final_engine_runtime_config','final_engine_source'];paths={k:project_path(cfg[k]) for k in keys};missing=[f'{k}: {p}' for k,p in paths.items() if not p.is_file()]
 if missing:raise FileNotFoundError('Missing authorization evidence:\n'+'\n'.join(missing))
 lock_before=paths['external_lock'].read_bytes();lock=json.loads(lock_before.decode('utf-8'));checksum=json.loads(paths['lock_checksum_record'].read_text(encoding='utf-8-sig'));runner=json.loads(paths['runner_preflight'].read_text(encoding='utf-8'));activation=json.loads(paths['activation_preflight'].read_text(encoding='utf-8'));engine=json.loads(paths['final_engine_preflight'].read_text(encoding='utf-8'));synthetic=json.loads(paths['final_engine_synthetic_integration'].read_text(encoding='utf-8'));runtime=yaml.safe_load(paths['final_engine_runtime_config'].read_text(encoding='utf-8-sig'));source=paths['final_engine_source'].read_text(encoding='utf-8')
 actual=changed_paths();allowed={str(p).replace('\\','/') for p in cfg['allowed_precommit_paths']};unexpected=sorted(actual-allowed);missing_expected=sorted(allowed-actual);guards=engine_ast_guards(source)
 checks={'review_only':args.review_only is True,'lock_sealed':lock.get('lock_status')=='sealed_pre_execution','execution_authorized':lock.get('execution_authorized') is True,'execution_count_zero':int(lock.get('execution_count',-1))==int(cfg['expected_execution_count']),'execution_limit_one':int(lock.get('execution_count_limit',-1))==int(cfg['execution_count_limit']),'slide_receipts_complete':len(lock.get('slides',[]))==int(cfg['expected_slides']),'lock_checksum_matches':hashlib.sha256(lock_before).hexdigest()==checksum.get('lock_sha256'),'git_commit_matches_lock':git('rev-parse','HEAD')==lock.get('git_commit'),'working_tree_contains_only_expected_review_files':actual==allowed and not unexpected and not missing_expected,'runner_ready':runner.get('runner_ready_for_one_time_execution') is True,'runner_non_consuming':runner.get('execution_count_consumed') is False and runner.get('inference_executed') is False,'activation_ready':activation.get('activation_contract_ready') is True,'activation_disabled':activation.get('activation_execution_enabled') is False,'engine_ready':engine.get('engine_ready') is True,'engine_execution_disabled':runtime.get('external_execution_enabled') is False,'real_wsi_access_disabled':runtime.get('real_wsi_access_enabled') is False,'engine_preflight_non_consuming':engine.get('execution_count_consumed') is False and engine.get('inference_executed') is False,'synthetic_integration_passed':synthetic.get('passed') is True,'synthetic_nonconsequential':synthetic.get('real_wsi_accessed') is False and synthetic.get('real_lock_mutated') is False and synthetic.get('external_execution_consumed') is False,'source_guards_complete':all(guards.values()),'review_prohibits_inference':cfg.get('prohibit_inference') is True,'review_prohibits_lock_mutation':cfg.get('prohibit_lock_mutation') is True,'review_prohibits_token_use':cfg.get('prohibit_execution_token_use') is True}
 checks['lock_unchanged_by_review']=paths['external_lock'].read_bytes()==lock_before;passed=all(checks.values());report={'schema_version':'1.2','project':cfg['project'],'scientific_scope':cfg['scientific_scope'],'review_only':True,'authorization_review_passed':passed,'external_execution_enabled':False,'real_wsi_access_enabled':False,'execution_token_used':False,'inference_executed':False,'execution_count_consumed':False,'working_tree_actual_paths':sorted(actual),'working_tree_allowed_paths':sorted(allowed),'unexpected_working_tree_paths':unexpected,'missing_expected_working_tree_paths':missing_expected,'source_guard_checks':guards,'checks':checks,'blocking_reasons':[k for k,v in checks.items() if not v],'git_commit':git('rev-parse','HEAD'),'external_lock_sha256':sha(paths['external_lock']),'config_sha256':sha(cp),'passed':passed}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'final_authorization_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
 if not passed:raise RuntimeError('Final authorization review failed')
 print('PASS: AST-validated final authorization review completed without enabling or consuming external execution.')
if __name__=='__main__':main()
