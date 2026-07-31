"""Aggregate repeated training-only OOF predictions to one record per slide."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np,yaml
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,roc_auc_score,average_precision_score
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_training_only_slide_level_calibration.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def load_csv(path):
 with path.open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def ece(y,p,bins):
 edges=np.linspace(0.,1.,bins+1);score=0.;rows=[]
 for i in range(bins):
  lo=float(edges[i]);hi=float(edges[i+1]);mask=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi));n=int(mask.sum())
  if n:mp=float(p[mask].mean());rate=float(y[mask].mean());gap=abs(mp-rate);score+=n/len(y)*gap
  else:mp=rate=gap=None
  rows.append({'bin_index':i,'lower':lo,'upper':hi,'count':n,'mean_probability':mp,'event_rate':rate,'absolute_gap':gap})
 return float(score),rows
def quantile(values,q):return float(np.quantile(np.asarray(values,dtype=np.float64),q))
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));paths={k:project_path(cfg[k]) for k in ['training_predictions','training_stability_results','row_level_calibration_results','disagreement_audit','frozen_validation_results','final_test_lock']}
 stability=json.loads(paths['training_stability_results'].read_text(encoding='utf-8'));rowcal=json.loads(paths['row_level_calibration_results'].read_text(encoding='utf-8'));disagreement=json.loads(paths['disagreement_audit'].read_text(encoding='utf-8'));validation=json.loads(paths['frozen_validation_results'].read_text(encoding='utf-8'))
 if any([stability['validation_slides_loaded']!=0,stability['test_slides_loaded']!=0,rowcal['validation_slides_loaded']!=0,rowcal['test_slides_loaded']!=0,disagreement['validation_slides_loaded']!=0,disagreement['test_slides_loaded']!=0,validation['test_slides_loaded']!=0]):raise RuntimeError('Upstream boundary invalid')
 if stability['validation_model_outputs_generated'] is not False or stability['test_model_outputs_generated'] is not False or rowcal['validation_model_outputs_generated'] is not False or rowcal['test_model_outputs_generated'] is not False or validation['test_model_outputs_generated'] is not False:raise RuntimeError('Upstream output boundary invalid')
 raw=load_csv(paths['training_predictions']);models=list(cfg['models']);selected=[r for r in raw if r['model_name'] in set(models)];expected=int(cfg['expected_training_slides'])*int(cfg['expected_repeats'])*len(models)
 if len(selected)!=expected:raise RuntimeError(f'Expected {expected} rows; found {len(selected)}')
 grouped=defaultdict(list)
 for r in selected:grouped[(r['model_name'],r['slide'])].append(r)
 slide_rows=[];model_results={};bin_rows=[];threshold=float(cfg['decision_threshold'])
 for model in models:
  model_slide=[]
  for slide in sorted({r['slide'] for r in selected}):
   rows=sorted(grouped[(model,slide)],key=lambda r:int(r['repeat']))
   if len(rows)!=int(cfg['expected_repeats']):raise RuntimeError(f'{model}/{slide}: incomplete repeats')
   probabilities=np.asarray([float(r['probability']) for r in rows]);label=int(rows[0]['label']);predictions=(probabilities>=threshold).astype(int);correct=int((predictions==label).sum());mean=float(probabilities.mean());median=float(np.median(probabilities));std=float(probabilities.std(ddof=1));record={'model_name':model,'slide':slide,'label':label,'repeat_count':len(rows),'mean_probability':mean,'median_probability':median,'probability_std':std,'probability_minimum':float(probabilities.min()),'probability_maximum':float(probabilities.max()),'probability_range':float(probabilities.max()-probabilities.min()),'q10_probability':quantile(probabilities,.1),'q25_probability':quantile(probabilities,.25),'q75_probability':quantile(probabilities,.75),'q90_probability':quantile(probabilities,.9),'correct_repeat_count':correct,'incorrect_repeat_count':len(rows)-correct,'mean_prediction':int(mean>=threshold),'median_prediction':int(median>=threshold),'mean_prediction_correct':int(mean>=threshold)==label,'median_prediction_correct':int(median>=threshold)==label,'persistent_correct':correct>=int(cfg['persistent_correct_minimum']),'persistent_incorrect':correct<=int(cfg['persistent_incorrect_maximum']),'split_sensitive':std>=float(cfg['split_sensitive_minimum_std'])}
   slide_rows.append(record);model_slide.append(record)
  y=np.asarray([r['label'] for r in model_slide]);p=np.asarray([r['mean_probability'] for r in model_slide]);pred=(p>=threshold).astype(int);ece_value,bins=ece(y,p,int(cfg['calibration_bins']))
  for row in bins:bin_rows.append({'model_name':model,**row})
  model_results[model]={'independent_slides':len(model_slide),'slide_level_brier_score':float(brier_score_loss(y,p)),'slide_level_expected_calibration_error':ece_value,'slide_level_accuracy':float(accuracy_score(y,pred)),'slide_level_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'slide_level_auroc':float(roc_auc_score(y,p)),'slide_level_auprc':float(average_precision_score(y,p)),'persistent_correct_slides':int(sum(r['persistent_correct'] for r in model_slide)),'persistent_incorrect_slides':int(sum(r['persistent_incorrect'] for r in model_slide)),'split_sensitive_slides':int(sum(r['split_sensitive'] for r in model_slide)),'mean_probability_std_across_slides':float(np.mean([r['probability_std'] for r in model_slide])),'calibration_bins':bins}
 # Cross-model slide comparison.
 index={(r['model_name'],r['slide']):r for r in slide_rows};comparison=[]
 for slide in sorted({r['slide'] for r in slide_rows}):
  a=index[(models[0],slide)];b=index[(models[1],slide)];comparison.append({'slide':slide,'label':a['label'],'primary_mean_probability':a['mean_probability'],'secondary_mean_probability':b['mean_probability'],'secondary_minus_primary_mean_probability':b['mean_probability']-a['mean_probability'],'absolute_mean_probability_difference':abs(b['mean_probability']-a['mean_probability']),'primary_correct_repeat_count':a['correct_repeat_count'],'secondary_correct_repeat_count':b['correct_repeat_count'],'secondary_minus_primary_correct_repeats':b['correct_repeat_count']-a['correct_repeat_count'],'primary_split_sensitive':a['split_sensitive'],'secondary_split_sensitive':b['split_sensitive'],'mean_prediction_disagreement':a['mean_prediction']!=b['mean_prediction']})
 summary={'schema_version':'1.0','dataset':cfg['dataset'],'scientific_scope':cfg['scientific_scope'],'models':models,'training_slides':int(cfg['expected_training_slides']),'repeats':int(cfg['expected_repeats']),'slide_level_records':len(slide_rows),'independent_slide_records_per_model':int(cfg['expected_training_slides']),'validation_slides_loaded':0,'test_slides_loaded':0,'model_outputs_generated':False,'validation_model_outputs_generated':False,'test_model_outputs_generated':False,'model_results':model_results,'slides_with_mean_prediction_disagreement':int(sum(r['mean_prediction_disagreement'] for r in comparison)),'training_predictions_sha256':sha(paths['training_predictions']),'training_stability_results_sha256':sha(paths['training_stability_results']),'row_level_calibration_results_sha256':sha(paths['row_level_calibration_results']),'disagreement_audit_sha256':sha(paths['disagreement_audit']),'frozen_validation_results_sha256':sha(paths['frozen_validation_results']),'final_test_lock_sha256':sha(paths['final_test_lock']),'config_sha256':sha(cp),'passed':True}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'training_only_slide_level_calibration.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 for name,data in [('slide_level_uncertainty.csv',slide_rows),('slide_level_calibration_bins.csv',bin_rows),('slide_level_model_comparison.csv',comparison)]:
  with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(json.dumps(summary,indent=2));print('PASS: Training-only slide-level uncertainty and calibration audit completed.')
if __name__=='__main__':main()
