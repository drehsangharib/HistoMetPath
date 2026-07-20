"""Build an annotation-independent consensus from frozen Spatial v2/v3 coordinates."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_v2_v3_consensus_development.yaml');return p.parse_args()
def farthest(coords,count,seeds=()):
 ordered=sorted(set(coords)); chosen=list(sorted(set(seeds)))
 if count<=0:return []
 if count>=len(ordered):return ordered
 remaining=set(ordered)
 if chosen:
  remaining-=set(chosen)
 else:
  chosen=[ordered[0]];remaining.remove(ordered[0])
 output=[]
 while len(output)<count and remaining:
  anchors=chosen+output
  pick=max(remaining,key=lambda p:(min((p[0]-q[0])**2+(p[1]-q[1])**2 for q in anchors),-p[1],-p[0]))
  output.append(pick);remaining.remove(pick)
 return output
def build_consensus(v2,v3,budget):
 a=set(v2);b=set(v3);shared=sorted(a&b);u2=sorted(a-b);u3=sorted(b-a)
 if len(shared)>=budget:
  selected=farthest(shared,budget)
  return selected,{'shared_total':len(shared),'v2_unique_total':len(u2),'v3_unique_total':len(u3),'shared_selected':len(selected),'v2_unique_selected':0,'v3_unique_selected':0}
 remaining=budget-len(shared);q2=remaining//2;q3=remaining-q2
 s2=farthest(u2,min(q2,len(u2)),shared);s3=farthest(u3,min(q3,len(u3)),shared+s2)
 spare=remaining-len(s2)-len(s3)
 if spare:
  r2=[p for p in u2 if p not in set(s2)];r3=[p for p in u3 if p not in set(s3)]
  pool=[('v2',p) for p in r2]+[('v3',p) for p in r3];anchors=shared+s2+s3
  while spare and pool:
   source,pick=max(pool,key=lambda x:(min((x[1][0]-q[0])**2+(x[1][1]-q[1])**2 for q in anchors),x[0],-x[1][1],-x[1][0]))
   (s2 if source=='v2' else s3).append(pick);anchors.append(pick);pool.remove((source,pick));spare-=1
 selected=sorted(shared+s2+s3)
 return selected,{'shared_total':len(shared),'v2_unique_total':len(u2),'v3_unique_total':len(u3),'shared_selected':len(shared),'v2_unique_selected':len(s2),'v3_unique_selected':len(s3)}
def main():
 args=parse_args();cfg=yaml.safe_load(project_path(args.config).read_text(encoding='utf-8-sig'));manifest=json.loads(project_path(cfg['processing_manifest']).read_text(encoding='utf-8'));rows=[r for r in manifest['slides'] if r['split'] in set(cfg['allowed_splits'])]
 if len(rows)!=36 or any(r['split']=='test' for r in rows):raise RuntimeError('development-only gate failed')
 r2=project_path(cfg['v2_coordinate_root']);r3=project_path(cfg['v3_coordinate_root']);out=project_path(cfg['output_root']);cr=out/'coordinates';out.mkdir(parents=True,exist_ok=True);cr.mkdir(parents=True,exist_ok=True);results=[]
 for row in sorted(rows,key=lambda r:r['slide']):
  v2=[tuple(map(int,x)) for x in np.load(r2/f"{row['slide']}_coordinates.npy",allow_pickle=False)];v3=[tuple(map(int,x)) for x in np.load(r3/f"{row['slide']}_coordinates.npy",allow_pickle=False)];selected,stats=build_consensus(v2,v3,int(cfg['max_tiles_per_slide']))
  np.save(cr/f"{row['slide']}_coordinates.npy",np.asarray(selected,dtype=np.int64),allow_pickle=False);results.append({'slide':row['slide'],'label':row['label'],'split':row['split'],'selected_tile_count':len(selected),**stats,'status':'complete'})
 payload={'schema_version':'1.0','dataset':cfg['dataset'],'development_slide_count':36,'test_slides_loaded':0,'model_outputs_generated':False,'slides':results,'passed':True};(out/'consensus_manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 with (out/'consensus_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 print(json.dumps({'development_slide_count':36,'test_slides_loaded':0,'mean_shared_selected':sum(r['shared_selected'] for r in results)/36,'passed':True},indent=2));print('PASS: Frozen development-only v2/v3 consensus sampler completed.')
if __name__=='__main__':main()
