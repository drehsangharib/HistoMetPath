"""Morphology-aware, annotation-independent Spatial Sampler v3."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np, openslide, yaml
from PIL import ImageFilter
from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler import spatial_bin
from core.wsi.run_camelyon16_spatial_sampler_v2 import allocate_density_budget
from core.wsi.tissue_mask import create_tissue_mask,tissue_fraction

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_spatial_sampler_v3_development.yaml');p.add_argument('--slides',nargs='+');p.add_argument('--force',action='store_true');return p.parse_args()
def descriptor(tile):
 a=np.asarray(tile,dtype=np.float32)/255.; hsv=np.asarray(tile.convert('HSV'),dtype=np.float32)/255.; edge=np.asarray(tile.convert('L').filter(ImageFilter.FIND_EDGES),dtype=np.float32)/255.
 optical=-np.log(np.clip(a,1/255.,1.)); return np.asarray([*a.mean((0,1)),*a.std((0,1)),hsv[:,:,1].mean(),hsv[:,:,1].std(),edge.mean(),edge.std(),optical.mean(),optical.std()],dtype=np.float32)
def normalize_matrix(values):
 values=np.asarray(values,dtype=np.float64); low=values.min(0); high=values.max(0); scale=high-low; scale[scale<1e-12]=1.; return (values-low)/scale
def joint_farthest_select(candidates,count,spatial_weight=1.,morphology_weight=1.):
 if count>=len(candidates): return sorted(candidates,key=lambda c:(-c['tissue_fraction'],c['y'],c['x']))
 spatial=normalize_matrix([[c['x'],c['y']] for c in candidates]); morph=normalize_matrix([c['descriptor'] for c in candidates]); features=np.concatenate([spatial*spatial_weight,morph*morphology_weight],1)
 order=sorted(range(len(candidates)),key=lambda i:(-candidates[i]['tissue_fraction'],candidates[i]['y'],candidates[i]['x'])); selected=[order[0]]; remaining=set(order[1:])
 while len(selected)<count:
  best=max(remaining,key=lambda i:(min(float(np.sum((features[i]-features[j])**2)) for j in selected),candidates[i]['tissue_fraction'],-candidates[i]['y'],-candidates[i]['x']))
  selected.append(best);remaining.remove(best)
 return [candidates[i] for i in selected]
def sample_slide(row,cfg):
 started=time.time();slide=openslide.OpenSlide(str(Path(row['path'])));bins={};examined=0
 try:
  level=int(row['selected_level']);down=float(row['selected_downsample']);lw,lh=slide.level_dimensions[level];sw,sh=slide.dimensions;ts=int(cfg['tile_size']);stride=int(cfg['stride']);gr=int(cfg['spatial_grid_rows']);gc=int(cfg['spatial_grid_columns'])
  for yl in range(0,lh-ts+1,stride):
   for xl in range(0,lw-ts+1,stride):
    examined+=1;x=int(round(xl*down));y=int(round(yl*down));tile=slide.read_region((x,y),level,(ts,ts)).convert('RGB');frac=tissue_fraction(create_tissue_mask(tile,intensity_threshold=int(cfg['intensity_threshold'])))
    if frac<float(cfg['min_tissue_fraction']):continue
    key=spatial_bin(x,y,sw,sh,gr,gc);bins.setdefault(key,[]).append({'x':x,'y':y,'tissue_fraction':float(frac),'descriptor':descriptor(tile).tolist(),'spatial_bin_row':key[0],'spatial_bin_column':key[1]})
 finally:slide.close()
 counts={k:len(v) for k,v in bins.items()};target=min(int(cfg['max_tiles_per_slide']),sum(counts.values()));alloc=allocate_density_budget(counts,target,int(cfg['minimum_tiles_per_occupied_bin']));selected=[]
 for key in sorted(bins):selected.extend(joint_farthest_select(bins[key],alloc[key],float(cfg['spatial_weight']),float(cfg['morphology_weight'])))
 selected.sort(key=lambda c:(c['spatial_bin_row'],c['spatial_bin_column'],c['y'],c['x']));coords=np.asarray([[c['x'],c['y']] for c in selected],dtype=np.int64);fractions=np.asarray([c['tissue_fraction'] for c in selected],dtype=np.float32)
 if len(selected)!=target:raise RuntimeError(f"{row['slide']}: budget mismatch")
 return {'slide':row['slide'],'label':row['label'],'split':row['split'],'selected_level':level,'selected_downsample':down,'effective_mpp':row['effective_mpp'],'candidates_examined':examined,'tissue_candidate_count':sum(counts.values()),'occupied_bin_count':len(bins),'selected_tile_count':len(selected),'budget_utilization_fraction':len(selected)/int(cfg['max_tiles_per_slide']),'coordinate_x_min':int(coords[:,0].min()),'coordinate_x_max':int(coords[:,0].max()),'coordinate_y_min':int(coords[:,1].min()),'coordinate_y_max':int(coords[:,1].max()),'mean_tissue_fraction':float(fractions.mean()),'elapsed_seconds':time.time()-started,'tiles':selected,'status':'complete'}
def write_csv(path,rows):
 flat=[{k:v for k,v in r.items() if k!='tiles'} for r in rows]
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));source=json.loads(project_path(cfg['processing_manifest']).read_text(encoding='utf-8'));allowed=set(cfg['allowed_splits']);rows=[r for r in source['slides'] if r['split'] in allowed]
 if len(rows)!=36 or any(r['split']=='test' for r in rows):raise RuntimeError('development split gate failed')
 if args.slides:
  requested=set(args.slides);rows=[r for r in rows if r['slide'] in requested];missing=requested-{r['slide'] for r in rows}
  if missing:raise RuntimeError(f'Unavailable or prohibited: {sorted(missing)}')
 out=project_path(cfg['output_root']);cr=out/'coordinates';out.mkdir(parents=True,exist_ok=True);cr.mkdir(parents=True,exist_ok=True);mp=out/'spatial_v3_sampling_manifest.json';prior={}
 if mp.is_file() and not args.force:prior={r['slide']:r for r in json.loads(mp.read_text(encoding='utf-8')).get('slides',[])}
 results=[]
 for i,row in enumerate(rows,1):
  if row['slide'] in prior and prior[row['slide']].get('status')=='complete':print(f"Skipping completed slide {row['slide']}");results.append(prior[row['slide']]);continue
  print(f"Spatial v3 sampling {i}/{len(rows)}: {row['slide']} ({row['split']})");result=sample_slide(row,cfg);results.append(result);np.save(cr/f"{row['slide']}_coordinates.npy",np.asarray([[t['x'],t['y']] for t in result['tiles']],dtype=np.int64),allow_pickle=False);np.save(cr/f"{row['slide']}_tissue_fractions.npy",np.asarray([t['tissue_fraction'] for t in result['tiles']],dtype=np.float32),allow_pickle=False)
  payload={'schema_version':'3.0','dataset':cfg['dataset'],'development_slide_count':len(rows),'test_slides_loaded':0,'model_outputs_generated':False,'slides':results,'completed_count':len(results),'passed':len(results)==len(rows)};mp.write_text(json.dumps(payload,indent=2),encoding='utf-8');write_csv(out/'spatial_v3_sampling_summary.csv',results)
 print(json.dumps({'development_slide_count':len(rows),'completed_count':len(results),'test_slides_loaded':0,'passed':len(results)==len(rows)},indent=2));print('PASS: Morphology-aware development-only Spatial Sampler v3 completed.')
if __name__=='__main__':main()
