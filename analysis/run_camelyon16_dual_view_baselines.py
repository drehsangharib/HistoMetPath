"""Development-only interpretable baselines for frozen v2/v3 embedding views."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import joblib,numpy as np,yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from core.wsi.run_camelyon16_batch_pipeline import project_path

LABELS={'normal':0,'tumor':1}
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_dual_view_baselines.yaml');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def select_threshold(y,p):
 candidates=sorted(set([0.0,1.0,*map(float,p)]));best=None
 for threshold in candidates:
  pred=(p>=threshold).astype(int);score=balanced_accuracy_score(y,pred);key=(score,-abs(threshold-.5),-threshold)
  if best is None or key>best[0]:best=(key,float(threshold))
 return best[1]
def metric_block(y,p,threshold):
 pred=(p>=threshold).astype(int);tn,fp,fn,tp=[int(x) for x in confusion_matrix(y,pred,labels=[0,1]).ravel()]
 return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'confusion_matrix':[[tn,fp],[fn,tp]]}
def fit_lr(x,y,c,max_iter,seed):
 scaler=StandardScaler().fit(x);model=LogisticRegression(C=c,max_iter=max_iter,random_state=seed).fit(scaler.transform(x),y);return scaler,model
def predict_lr(bundle,x):
 scaler,model=bundle;return model.predict_proba(scaler.transform(x))[:,1]
def build_dataset(cfg):
 manifest_path=project_path(cfg['dual_view_manifest']);manifest=json.loads(manifest_path.read_text(encoding='utf-8'));records=manifest['records']
 if manifest.get('test_slides_loaded')!=0 or manifest.get('model_outputs_generated') is not False:raise RuntimeError('Dual-view input boundary invalid')
 grouped={}
 for record in records:
  if record['split'] not in set(cfg['allowed_splits']):raise RuntimeError(f"Prohibited split: {record['split']}")
  grouped.setdefault(record['slide'],{'slide':record['slide'],'label':record['label'],'split':record['split']})[record['view']]=record
 rows=[];root=project_path(cfg['embedding_root'])
 for slide in sorted(grouped):
  item=grouped[slide]
  if set(cfg['views'])-set(item):raise RuntimeError(f'{slide}: missing view')
  row={'slide':slide,'label_text':item['label'],'label':LABELS[item['label']],'split':item['split']}
  for view in cfg['views']:
   path=root/view/f'{slide}_embeddings.npy';array=np.load(path,allow_pickle=False)
   if array.shape!=(300,512) or not np.isfinite(array).all():raise RuntimeError(f'{view}/{slide}: invalid embeddings')
   row[view]=array.mean(0).astype(np.float64)
  rows.append(row)
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
 if len(train)!=30 or len(val)!=6:raise RuntimeError(f'Expected 30/6 train/validation; got {len(train)}/{len(val)}')
 return train,val,manifest_path

def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));train,val,manifest_path=build_dataset(cfg);seed=int(cfg['seed']);lr=cfg['logistic_regression'];y=np.asarray([r['label'] for r in train]);yv=np.asarray([r['label'] for r in val]);views=cfg['views']
 x={v:np.stack([r[v] for r in train]) for v in views};xv={v:np.stack([r[v] for r in val]) for v in views};results=[];artifacts={};view_probs={}
 for view in views:
  bundle=fit_lr(x[view],y,float(lr['c']),int(lr['max_iter']),seed);prob=predict_lr(bundle,xv[view]);threshold=select_threshold(yv,prob);view_probs[view]=prob;artifacts[f'{view}_mean_pool_lr']={'type':'single_view','view':view,'scaler':bundle[0],'model':bundle[1]};results.append((f'{view}_mean_pool_lr',prob,threshold))
 concat=np.concatenate([x[v] for v in views],1);concatv=np.concatenate([xv[v] for v in views],1);bundle=fit_lr(concat,y,float(lr['c']),int(lr['max_iter']),seed);prob=predict_lr(bundle,concatv);threshold=select_threshold(yv,prob);artifacts['dual_view_mean_concat_lr']={'type':'concat','views':views,'scaler':bundle[0],'model':bundle[1]};results.append(('dual_view_mean_concat_lr',prob,threshold))
 average=np.mean(np.stack([view_probs[v] for v in views]),0);results.append(('dual_view_late_probability_average',average,select_threshold(yv,average)));artifacts['dual_view_late_probability_average']={'type':'average','source_models':[f'{v}_mean_pool_lr' for v in views]}
 folds=int(cfg['stacking']['folds']);skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed);oof=np.zeros((len(train),len(views)));base_full={}
 for vi,view in enumerate(views):
  for tr,te in skf.split(x[view],y):
   fold=fit_lr(x[view][tr],y[tr],float(lr['c']),int(lr['max_iter']),seed);oof[te,vi]=predict_lr(fold,x[view][te])
  base_full[view]=fit_lr(x[view],y,float(lr['c']),int(lr['max_iter']),seed)
 meta=fit_lr(oof,y,float(lr['c']),int(lr['max_iter']),seed);val_meta=np.column_stack([predict_lr(base_full[v],xv[v]) for v in views]);prob=predict_lr(meta,val_meta);results.append(('dual_view_oof_logistic_stacking',prob,select_threshold(yv,prob)));artifacts['dual_view_oof_logistic_stacking']={'type':'stacking','views':views,'folds':folds,'base_models':base_full,'meta_scaler':meta[0],'meta_model':meta[1]}
 output=[]
 for name,prob,threshold in results:
  output.append({'model_name':name,'threshold':float(threshold),'validation_metrics':metric_block(yv,prob,threshold),'validation_predictions':[{'slide':r['slide'],'label':r['label_text'],'probability':float(p),'prediction':int(p>=threshold)} for r,p in zip(val,prob)]})
 def key(r):
  m=r['validation_metrics'];return (m['balanced_accuracy'],m['auroc'],m['auprc'],r['model_name'])
 selected=max(output,key=key);out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);artifact_path=out/'dual_view_baseline_artifacts.joblib';joblib.dump({'schema_version':'1.0','artifacts':artifacts,'selected_model':selected['model_name'],'selected_threshold':selected['threshold']},artifact_path)
 report={'schema_version':'1.0','dataset':cfg['dataset'],'scientific_scope':cfg['scientific_scope'],'train_slides':30,'validation_slides':6,'test_slides_loaded':0,'model_outputs_generated':True,'test_model_outputs_generated':False,'models_compared':len(output),'results':output,'selected_model':selected['model_name'],'selected_threshold':selected['threshold'],'dual_view_manifest_sha256':sha(manifest_path),'config_sha256':sha(cp),'artifact_sha256':sha(artifact_path),'final_test_lock_sha256':sha(project_path(cfg['final_test_lock'])),'passed':True};(out/'dual_view_baseline_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 rows=[]
 for result in output:
  rows.append({'model_name':result['model_name'],'threshold':result['threshold'],**result['validation_metrics']})
 with (out/'dual_view_baseline_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:v for k,v in report.items() if k not in {'results'}},indent=2));print('PASS: Development-only dual-view baseline comparison completed.')
if __name__=='__main__':main()
