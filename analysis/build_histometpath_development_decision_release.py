"""Build the frozen HistoMetPath internal-development decision release."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/release/histometpath_development_decision.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));evidence_paths={name:project_path(value) for name,value in cfg['required_evidence'].items()}
 missing=[str(path) for path in evidence_paths.values() if not path.is_file()]
 if missing:raise FileNotFoundError(f'Missing required evidence: {missing}')
 final_lock=load(evidence_paths['final_test_lock']);final_result=load(evidence_paths['final_test_result']);portfolio=load(evidence_paths['sampler_portfolio']);loss=load(evidence_paths['consensus_loss_attribution']);baseline=load(evidence_paths['dual_view_baselines']);stability=load(evidence_paths['training_stability']);disagreement=load(evidence_paths['disagreement_audit']);calibration=load(evidence_paths['calibration_selective']);slidecal=load(evidence_paths['slide_level_calibration'])
 gates={
  'final_test_executed_once': final_lock.get('executed_once') is True,
  'sampler_portfolio_test_slides_zero': portfolio.get('test_slides_loaded')==0,
  'training_stability_validation_slides_zero': stability.get('validation_slides_loaded')==0,
  'training_stability_test_slides_zero': stability.get('test_slides_loaded')==0,
  'disagreement_validation_slides_zero': disagreement.get('validation_slides_loaded')==0,
  'disagreement_test_slides_zero': disagreement.get('test_slides_loaded')==0,
  'calibration_validation_slides_zero': calibration.get('validation_slides_loaded')==0,
  'calibration_test_slides_zero': calibration.get('test_slides_loaded')==0,
  'slide_calibration_validation_slides_zero': slidecal.get('validation_slides_loaded')==0,
  'slide_calibration_test_slides_zero': slidecal.get('test_slides_loaded')==0,
 }
 if not all(gates.values()):raise RuntimeError(f'Boundary gate failed: {gates}')
 v2=portfolio['sampler_metrics']['spatial_v2'];v3=portfolio['sampler_metrics']['spatial_v3'];slide_v2=slidecal['model_results']['spatial_v2_mean_pool_lr'];slide_concat=slidecal['model_results']['dual_view_mean_concat_lr']
 report={
  'schema_version':'1.0','project':cfg['project'],'release_scope':cfg['release_scope'],'status':'internal_development_cycle_complete','scientific_boundary_gates':gates,'frozen_decisions':cfg['frozen_decisions'],
  'sampler_evidence':{'pareto_front':portfolio['pareto_front'],'spatial_v2_lesion_bags':round(v2['lesion_positive_bag_fraction']*18),'spatial_v3_lesion_bags':round(v3['lesion_positive_bag_fraction']*18),'spatial_v2_lesion_tiles':v2['total_lesion_intersecting_tiles'],'spatial_v3_lesion_tiles':v3['total_lesion_intersecting_tiles'],'union_upper_bound_lesion_bags':loss['union_upper_bound_bags_with_lesion']},
  'frozen_validation_evidence':{'selected_model':baseline['selected_model'],'validation_slides':baseline['validation_slides'],'test_slides_loaded':baseline['test_slides_loaded']},
  'training_stability_evidence':{'selected_model':stability['training_only_selected_model'],'total_outer_folds':stability['total_outer_folds']},
  'slide_level_evidence':{'spatial_v2':slide_v2,'dual_view_mean_concat':slide_concat,'mean_prediction_disagreements':slidecal['slides_with_mean_prediction_disagreement']},
  'calibration_evidence':{'agreement_coverage':calibration['agreement_policy']['agreement_coverage'],'agreement_accuracy':calibration['agreement_policy']['agreement_accuracy'],'raw_probabilities_are_calibrated_risk':False},
  'historical_final_test':{'immutable':True,'result':final_result,'must_not_be_rerun_or_used_for_tuning':True},
  'required_next_external_step':'Establish a new untouched evaluation cohort before any new performance claim.',
  'evidence_sha256':{name:sha(path) for name,path in evidence_paths.items()},'config_sha256':sha(cp),'passed':True}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'development_decision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 lines=['# HistoMetPath Frozen Development Decision','','## Status','Internal CAMELYON16 development cycle complete.','','## Frozen decision',f"- Primary development baseline: `{cfg['frozen_decisions']['primary_development_baseline']}`",f"- Secondary exploratory candidate: `{cfg['frozen_decisions']['secondary_exploratory_candidate']}`",'- Raw probabilities are not calibrated tumor-risk estimates.','- OOF stacking and abstention policies are discontinued.','- The completed final test is immutable and must never be rerun or used for tuning.','','## Evidence summary',f"- Sampler Pareto front: {', '.join(portfolio['pareto_front'])}",f"- Spatial v2 lesion-positive bags: {round(v2['lesion_positive_bag_fraction']*18)}/18",f"- Spatial v3 lesion-positive bags: {round(v3['lesion_positive_bag_fraction']*18)}/18",f"- Parent-union upper bound: {loss['union_upper_bound_bags_with_lesion']}/18",f"- Frozen validation selection: {baseline['selected_model']}",f"- Training-only stability selection: {stability['training_only_selected_model']}",f"- Slide-level Spatial v2 AUROC/AUPRC: {slide_v2['slide_level_auroc']:.4f}/{slide_v2['slide_level_auprc']:.4f}",f"- Slide-level concatenation AUROC/AUPRC: {slide_concat['slide_level_auroc']:.4f}/{slide_concat['slide_level_auprc']:.4f}",'','## Required next step','A new untouched evaluation cohort is required before any new comparative performance claim.','']
 (out/'DEVELOPMENT_DECISION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'status':report['status'],'primary':cfg['frozen_decisions']['primary_development_baseline'],'secondary':cfg['frozen_decisions']['secondary_exploratory_candidate'],'boundary_gates_passed':all(gates.values()),'passed':True},indent=2));print('PASS: Frozen development decision release built.')
if __name__=='__main__':main()
