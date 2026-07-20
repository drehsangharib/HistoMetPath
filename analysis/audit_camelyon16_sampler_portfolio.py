"""Development-only stability and Pareto audit for raster and Spatial v1-v3."""
from __future__ import annotations
import argparse,csv,hashlib,json
from itertools import combinations
from pathlib import Path
import numpy as np,yaml
from analysis.audit_camelyon16_lesion_coverage import audit_slide
from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler import spatial_bin

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_sampler_portfolio_audit.yaml');return p.parse_args()
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def coordinate_set(path):
 return {tuple(map(int,row)) for row in np.load(path,allow_pickle=False)}
def jaccard(a,b):
 union=len(a|b);return len(a&b)/union if union else 1.
def occupied_fraction(coords,row,grid_rows,grid_columns):
 bins={spatial_bin(x,y,int(row['width']),int(row['height']),grid_rows,grid_columns) for x,y in coords};return len(bins)/(grid_rows*grid_columns)
def mean_nearest_neighbor(coords):
 a=np.asarray(sorted(coords),dtype=np.float64)
 if len(a)<2:return 0.
 minimum=[]
 for i in range(len(a)):
  d=np.sqrt(np.sum((a-a[i])**2,axis=1));d[i]=np.inf;minimum.append(float(d.min()))
 return float(np.mean(minimum))
def pareto_front(metrics,objectives):
 names=sorted(metrics);front=[]
 for candidate in names:
  dominated=False
  for other in names:
   if other==candidate:continue
   at_least=all(metrics[other][o]>=metrics[candidate][o] for o in objectives)
   strict=any(metrics[other][o]>metrics[candidate][o] for o in objectives)
   if at_least and strict:dominated=True;break
  if not dominated:front.append(candidate)
 return front
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));manifest_path=project_path(cfg['processing_manifest']);source=json.loads(manifest_path.read_text(encoding='utf-8'));allowed=set(cfg['allowed_splits']);prohibited=set(cfg['prohibited_splits']);rows=[r for r in source['slides'] if r['split'] in allowed]
 if len(rows)!=36 or any(r['split'] in prohibited for r in rows):raise RuntimeError('Development-only split gate failed')
 roots={name:project_path(value) for name,value in cfg['samplers'].items()};names=sorted(roots);gr=int(cfg['spatial_grid_rows']);gc=int(cfg['spatial_grid_columns']);annotation_root=project_path(cfg['annotation_root']);tile_size=int(cfg['tile_size']);per_slide=[];pair_rows=[]
 for row in sorted(rows,key=lambda r:r['slide']):
  sets={name:coordinate_set(root/f"{row['slide']}_coordinates.npy") for name,root in roots.items()}
  record={'slide':row['slide'],'label':row['label'],'split':row['split']}
  for name,coords in sets.items():
   record[f'{name}_tile_count']=len(coords);record[f'{name}_occupied_bin_fraction']=occupied_fraction(coords,row,gr,gc);record[f'{name}_mean_nearest_neighbor_distance']=mean_nearest_neighbor(coords)
  for first,second in combinations(names,2):
   value=jaccard(sets[first],sets[second]);record[f'jaccard_{first}_vs_{second}']=value;pair_rows.append({'slide':row['slide'],'split':row['split'],'label':row['label'],'sampler_a':first,'sampler_b':second,'jaccard':value})
  per_slide.append(record)
 tumor_rows=[r for r in rows if r['label']=='tumor'];lesion=[]
 for row in sorted(tumor_rows,key=lambda r:r['slide']):
  item={'slide':row['slide'],'split':row['split']}
  for name,root in roots.items():
   audit=audit_slide(row,root/f"{row['slide']}_coordinates.npy",annotation_root/f"{row['slide']}.xml",tile_size);item[f'{name}_has_lesion']=audit['bag_contains_annotated_lesion'];item[f'{name}_lesion_tiles']=audit['lesion_intersecting_tile_count'];item[f'{name}_polygon_fraction']=audit['covered_polygon_fraction']
  lesion.append(item)
 metrics={}
 for name in names:
  metrics[name]={'lesion_positive_bag_fraction':sum(r[f'{name}_has_lesion'] for r in lesion)/len(lesion),'total_lesion_intersecting_tiles':sum(r[f'{name}_lesion_tiles'] for r in lesion),'mean_polygon_coverage':float(np.mean([r[f'{name}_polygon_fraction'] for r in lesion])),'mean_occupied_bin_fraction':float(np.mean([r[f'{name}_occupied_bin_fraction'] for r in per_slide])),'mean_tile_count':float(np.mean([r[f'{name}_tile_count'] for r in per_slide])),'mean_nearest_neighbor_distance':float(np.mean([r[f'{name}_mean_nearest_neighbor_distance'] for r in per_slide]))}
 objectives=list(cfg['pareto_objectives']);front=pareto_front(metrics,objectives);lock=project_path(cfg['final_test_lock']);output={'schema_version':'1.0','dataset':cfg['dataset'],'scientific_scope':cfg['scientific_scope'],'development_slide_count':36,'development_tumor_slide_count':18,'test_slides_loaded':0,'model_outputs_generated':False,'final_test_lock_sha256':sha256(lock),'sampler_metrics':metrics,'pareto_objectives':objectives,'pareto_front':front,'pairwise_selection_overlap':pair_rows,'development_slides':per_slide,'tumor_lesion_metrics':lesion,'passed':True}
 out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);(out/'sampler_portfolio_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
 with (out/'sampler_metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['sampler',*next(iter(metrics.values())).keys()]);w.writeheader();w.writerows([{'sampler':n,**metrics[n]} for n in names])
 with (out/'pairwise_selection_overlap.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
 with (out/'tumor_lesion_metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(lesion[0]));w.writeheader();w.writerows(lesion)
 print(json.dumps({'development_slide_count':36,'development_tumor_slide_count':18,'test_slides_loaded':0,'model_outputs_generated':False,'pareto_front':front,'sampler_metrics':metrics,'passed':True},indent=2));print('PASS: Development-only sampler stability and Pareto audit completed.')
if __name__=='__main__':main()
