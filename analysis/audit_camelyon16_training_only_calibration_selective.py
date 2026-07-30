"""Training-only calibration and selective-prediction audit.

Uses existing repeated out-of-fold predictions for the frozen Spatial v2
primary baseline and dual-view concatenation secondary candidate. No model is
refit and no validation/test probabilities are loaded or generated.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,yaml
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_training_only_calibration_selective.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def load_csv(path):
 with path.open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def ece(y,p,bins):
 edges=np.linspace(0.,1.,bins+1);rows=[];total=len(y);score=0.
 for index in range(bins):
  lower=float(edges[index]);upper=float(edges[index+1]);mask=(p>=lower)&((p<upper) if index<bins-1 else (p<=upper));count=int(mask.sum())
  if count:
   mean_probability=float(p[mask].mean());event_rate=float(y[mask].mean());gap=abs(mean_probability-event_rate);score+=count/total*gap
  else:mean_probability=None;event_rate=None;gap=None
  rows.append({'bin_index':index,'lower':lower,'upper':upper,'count':count,'mean_probability':mean_probability,'event_rate':event_rate,'absolute_gap':gap})
 return float(score),rows
def selective_rows(y,p,slides,levels,threshold):
 confidence=np.abs(p-.5)*2.;order=np.lexsort((slides,-confidence));rows=[]
 for level in levels:
  retain=max(1,int(np.ceil(len(y)*float(level))));idx=order[:retain];pred=(p[idx]>=threshold).astype(int);labels=y[idx]
  rows.append({'requested_coverage':float(level),'retained_count':int(retain),'actual_coverage':float(retain/len(y)),'accuracy':float(accuracy_score(labels,pred)),'balanced_accuracy':float(balanced_accuracy_score(labels,pred)) if len(np.unique(labels))==2 else None,'mean_confidence':float(confidence[idx].mean()),'abstained_count':int(len(y)-retain)})
 return rows
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));pred_path=project_path(cfg['training_predictions']);stability_path=project_path(cfg['training_stability_results']);disagreement_path=project_path(cfg['disagreement_audit']);validation_path=project_path(cfg['frozen_validation_results']);lock_path=project_path(cfg['final_test_lock'])
 stability=json.loads(stability_path.read_text(encoding='utf-8'));disagreement=json.loads(disagreement_path.read_text(encoding='utf-8'));validation=json.loads(validation_path.read_text(encoding='utf-8'))
 if stability['validation_slides_loaded']!=0 or stability['test_slides_loaded']!=0:raise RuntimeError('Stability boundary invalid')
 if stability['validation_model_outputs_generated'] is not False or stability['test_model_outputs_generated'] is not False:raise RuntimeError('Stability output boundary invalid')
 if disagreement['validation_slides_loaded']!=0 or disagreement['test_slides_loaded']!=0:raise RuntimeError('Disagreement boundary invalid')
 if validation['test_slides_loaded']!=0 or validation['test_model_outputs_generated'] is not False:raise RuntimeError('Frozen validation boundary invalid')
 raw=load_csv(pred_path);models=list(cfg['models']);selected=[r for r in raw if r['model_name'] in set(models)];expected=int(cfg['expected_training_slides'])*int(cfg['expected_repeats'])*len(models)
 if len(selected)!=expected:raise RuntimeError(f'Expected {expected} prediction rows; found {len(selected)}')
 results={};calibration_rows=[];selective=[];threshold=float(cfg['decision_threshold']);bins=int(cfg['calibration_bins'])
 for model in models:
  rows=[r for r in selected if r['model_name']==model];rows.sort(key=lambda r:(int(r['repeat']),r['slide']));y=np.asarray([int(r['label']) for r in rows]);p=np.asarray([float(r['probability']) for r in rows]);slides=np.asarray([r['slide'] for r in rows]);pred=(p>=threshold).astype(int);ece_value,bin_rows=ece(y,p,bins)
  for row in bin_rows:calibration_rows.append({'model_name':model,**row})
  curves=selective_rows(y,p,slides,cfg['coverage_levels'],threshold)
  for row in curves:selective.append({'model_name':model,**row})
  results[model]={'prediction_rows':len(rows),'brier_score':float(brier_score_loss(y,p)),'expected_calibration_error':ece_value,'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'mean_confidence':float((np.abs(p-.5)*2.).mean()),'calibration_bins':bin_rows,'selective_prediction':curves}
 # Agreement-based selective policy: retain rows where both models agree, ranked by minimum confidence.
 keyed={}
 for r in selected:keyed[(r['model_name'],r['slide'],int(r['repeat']))]=r
 agreement=[]
 for slide in sorted({r['slide'] for r in selected}):
  for repeat in range(int(cfg['expected_repeats'])):
   a=keyed[(models[0],slide,repeat)];b=keyed[(models[1],slide,repeat)];pa=float(a['probability']);pb=float(b['probability']);pred_a=int(pa>=threshold);pred_b=int(pb>=threshold);label=int(a['label']);agreement.append({'slide':slide,'repeat':repeat,'label':label,'primary_probability':pa,'secondary_probability':pb,'agree':pred_a==pred_b,'agreed_prediction':pred_a if pred_a==pred_b else None,'agreement_correct':(pred_a==label) if pred_a==pred_b else None,'minimum_confidence':min(abs(pa-.5)*2.,abs(pb-.5)*2.)})
 agreed=[r for r in agreement if r['agree']];agreement_accuracy=float(np.mean([r['agreement_correct'] for r in agreed])) if agreed else None
 summary={'schema_version':'1.0','dataset':cfg['dataset'],'scientific_scope':cfg['scientific_scope'],'models':models,'training_slides':int(cfg['expected_training_slides']),'repeats':int(cfg['expected_repeats']),'prediction_rows':len(selected),'validation_slides_loaded':0,'test_slides_loaded':0,'model_outputs_generated':False,'validation_model_outputs_generated':False,'test_model_outputs_generated':False,'decision_threshold':threshold,'calibration_bins':bins,'model_results':results,'agreement_policy':{'paired_rows':len(agreement),'agreed_rows':len(agreed),'agreement_coverage':float(len(agreed)/len(agreement)),'agreement_accuracy':agreement_accuracy,'disagreement_rows':len(agreement)-len(agreed)},'training_predictions_sha256':sha(pred_path),'training_stability_results_sha256':sha(stability_path),'disagreement_audit_sha256':sha(disagreement_path),'frozen_validation_results_sha256':sha(validation_path),'final_test_lock_sha256':sha(lock_path),'config_sha256':sha(cp),'passed':True}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'training_only_calibration_selective.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 for name,data in [('calibration_bins.csv',calibration_rows),('selective_prediction_curves.csv',selective),('agreement_policy_rows.csv',agreement)]:
  with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(json.dumps({k:v for k,v in summary.items() if k!='model_results'}|{'model_results':results},indent=2));print('PASS: Training-only calibration and selective-prediction audit completed.')
if __name__=='__main__':main()
