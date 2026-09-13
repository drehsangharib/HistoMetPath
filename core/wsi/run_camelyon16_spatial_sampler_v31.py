"""Spatial-v3.1 operational-control instrumentation.

Preserves the canonical Spatial-v3 sampling algorithm while adding exact read
accounting, final-array validation, atomic per-slide promotion, receipts, and
restart-safe reuse. This is not a historical byte-identical replay.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,tempfile,time
from pathlib import Path
import math
import numpy as np, openslide, yaml
from PIL import ImageFilter
def project_path(value):
    """Resolve a project-relative path without importing the torch-based batch pipeline."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path
def spatial_bin(
    x_zero: int,
    y_zero: int,
    slide_width: int,
    slide_height: int,
    grid_rows: int,
    grid_columns: int,
) -> tuple[int, int]:
    column = min(grid_columns - 1, int(x_zero * grid_columns / slide_width))
    row = min(grid_rows - 1, int(y_zero * grid_rows / slide_height))
    return row, column
def allocate_density_budget(counts: dict, budget: int, minimum: int=1) -> dict:
    keys=sorted(counts)
    if not keys or budget<=0: return {key:0 for key in keys}
    allocation={key:min(minimum,counts[key]) for key in keys}
    used=sum(allocation.values())
    if used>budget:
        return {key:(1 if index<budget else 0) for index,key in enumerate(keys)}
    remaining=budget-used
    capacities={key:max(0,counts[key]-allocation[key]) for key in keys}
    while remaining>0 and sum(capacities.values())>0:
        total_capacity=sum(capacities.values())
        quotas={key:remaining*capacities[key]/total_capacity for key in keys}
        floors={key:min(capacities[key],int(math.floor(quotas[key]))) for key in keys}
        floor_total=sum(floors.values())
        if floor_total:
            for key in keys:
                allocation[key]+=floors[key]; capacities[key]-=floors[key]
            remaining-=floor_total
            if remaining<=0: break
        ranked=sorted(keys,key=lambda key:(-(quotas[key]-math.floor(quotas[key])),-capacities[key],key))
        changed=False
        for key in ranked:
            if remaining<=0: break
            if capacities[key]>0:
                allocation[key]+=1; capacities[key]-=1; remaining-=1; changed=True
        if not changed: break
    return allocation
from core.wsi.tissue_mask import create_tissue_mask,tissue_fraction

CANONICAL_PARENT_SHA256="18b58cb2571d0e81f213a399a07933e59e493dd2cb8e40ce47a982be487ec87f"
SCHEMA_VERSION="3.1"

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--slides',nargs='+');p.add_argument('--force',action='store_true');p.add_argument('--execute-spatial-v31',action='store_true');return p.parse_args()

def descriptor(tile):
 a=np.asarray(tile,dtype=np.float32)/255.;hsv=np.asarray(tile.convert('HSV'),dtype=np.float32)/255.;edge=np.asarray(tile.convert('L').filter(ImageFilter.FIND_EDGES),dtype=np.float32)/255.
 optical=-np.log(np.clip(a,1/255.,1.));return np.asarray([*a.mean((0,1)),*a.std((0,1)),hsv[:,:,1].mean(),hsv[:,:,1].std(),edge.mean(),edge.std(),optical.mean(),optical.std()],dtype=np.float32)

def normalize_matrix(values):
 values=np.asarray(values,dtype=np.float64);low=values.min(0);high=values.max(0);scale=high-low;scale[scale<1e-12]=1.;return (values-low)/scale

def joint_farthest_select(candidates,count,spatial_weight=1.,morphology_weight=1.):
 if count>=len(candidates):return sorted(candidates,key=lambda c:(-c['tissue_fraction'],c['y'],c['x']))
 spatial=normalize_matrix([[c['x'],c['y']] for c in candidates]);morph=normalize_matrix([c['descriptor'] for c in candidates]);features=np.concatenate([spatial*spatial_weight,morph*morphology_weight],1)
 order=sorted(range(len(candidates)),key=lambda i:(-candidates[i]['tissue_fraction'],candidates[i]['y'],candidates[i]['x']));selected=[order[0]];remaining=set(order[1:])
 while len(selected)<count:
  best=max(remaining,key=lambda i:(min(float(np.sum((features[i]-features[j])**2)) for j in selected),candidates[i]['tissue_fraction'],-candidates[i]['y'],-candidates[i]['x']))
  selected.append(best);remaining.remove(best)
 return [candidates[i] for i in selected]

def validate_arrays(coords,fractions,expected_count):
 coords=np.asarray(coords);fractions=np.asarray(fractions)
 if coords.dtype!=np.int64 or coords.shape!=(expected_count,2):raise RuntimeError(f'coordinate schema mismatch: {coords.dtype} {coords.shape}')
 if fractions.dtype!=np.float32 or fractions.shape!=(expected_count,):raise RuntimeError(f'tissue-fraction schema mismatch: {fractions.dtype} {fractions.shape}')
 unique=np.unique(coords,axis=0)
 if len(unique)!=expected_count:raise RuntimeError(f'duplicate coordinates after int64 conversion: unique={len(unique)} expected={expected_count}')
 if not np.isfinite(fractions).all():raise RuntimeError('non-finite tissue fractions')
 return {'coordinate_count':expected_count,'unique_coordinate_count':len(unique),'duplicate_coordinate_count':0}

def sample_slide(row,cfg,slide_factory=openslide.OpenSlide):
 started=time.time();slide=slide_factory(str(Path(row['path'])));bins={};examined=0;region_reads=0;successful_region_reads=0
 try:
  level=int(row['selected_level']);down=float(row['selected_downsample']);lw,lh=slide.level_dimensions[level];sw,sh=slide.dimensions;ts=int(cfg['tile_size']);stride=int(cfg['stride']);gr=int(cfg['spatial_grid_rows']);gc=int(cfg['spatial_grid_columns'])
  for yl in range(0,lh-ts+1,stride):
   for xl in range(0,lw-ts+1,stride):
    examined+=1;x=int(round(xl*down));y=int(round(yl*down));region_reads+=1
    tile=slide.read_region((x,y),level,(ts,ts)).convert('RGB');successful_region_reads+=1
    frac=tissue_fraction(create_tissue_mask(tile,intensity_threshold=int(cfg['intensity_threshold'])))
    if frac<float(cfg['min_tissue_fraction']):continue
    key=spatial_bin(x,y,sw,sh,gr,gc);bins.setdefault(key,[]).append({'x':x,'y':y,'tissue_fraction':float(frac),'descriptor':descriptor(tile).tolist(),'spatial_bin_row':key[0],'spatial_bin_column':key[1]})
 finally:slide.close()
 counts={k:len(v) for k,v in bins.items()};target=min(int(cfg['max_tiles_per_slide']),sum(counts.values()));alloc=allocate_density_budget(counts,target,int(cfg['minimum_tiles_per_occupied_bin']));selected=[]
 for key in sorted(bins):selected.extend(joint_farthest_select(bins[key],alloc[key],float(cfg['spatial_weight']),float(cfg['morphology_weight'])))
 selected.sort(key=lambda c:(c['spatial_bin_row'],c['spatial_bin_column'],c['y'],c['x']));coords=np.asarray([[c['x'],c['y']] for c in selected],dtype=np.int64);fractions=np.asarray([c['tissue_fraction'] for c in selected],dtype=np.float32)
 if len(selected)!=target:raise RuntimeError(f"{row['slide']}: budget mismatch")
 validation=validate_arrays(coords,fractions,target)
 return {'slide':row['slide'],'label':row['label'],'split':row['split'],'selected_level':level,'selected_downsample':down,'effective_mpp':row['effective_mpp'],'candidates_examined':examined,'region_reads':region_reads,'successful_region_reads':successful_region_reads,'tissue_candidate_count':sum(counts.values()),'occupied_bin_count':len(bins),'selected_tile_count':len(selected),'budget_utilization_fraction':len(selected)/int(cfg['max_tiles_per_slide']),'coordinate_x_min':int(coords[:,0].min()),'coordinate_x_max':int(coords[:,0].max()),'coordinate_y_min':int(coords[:,1].min()),'coordinate_y_max':int(coords[:,1].max()),'mean_tissue_fraction':float(fractions.mean()),'elapsed_seconds':time.time()-started,'deterministic_execution':True,'seed':None,'validation':validation,'tiles':selected,'status':'complete'},coords,fractions

def paths_for(root,slide_id):
 d=Path(root)/slide_id;return d,d/f'{slide_id}_coordinates.npy',d/f'{slide_id}_tissue_fraction.npy',d/f'{slide_id}_spatial_v31_receipt.json'

def validate_restart(root,row,expected_count):
 d,cp,fp,rp=paths_for(root,row['slide'])
 if not all(p.is_file() for p in (cp,fp,rp)):return None
 rec=json.loads(rp.read_text(encoding='utf-8-sig'))
 if rec.get('slide')!=row['slide'] or rec.get('label')!=row['label'] or rec.get('source_path')!=str(Path(row['path'])):return None
 if rec.get('coordinate_sha256')!=sha256_file(cp) or rec.get('tissue_fraction_sha256')!=sha256_file(fp):return None
 coords=np.load(cp,allow_pickle=False);fractions=np.load(fp,allow_pickle=False);validate_arrays(coords,fractions,expected_count)
 return rec

def atomic_write_slide(root,row,result,coords,fractions):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);final_dir,_,_,_=paths_for(root,row['slide'])
 if final_dir.exists():raise RuntimeError(f'final slide directory already exists: {final_dir}')
 with tempfile.TemporaryDirectory(prefix=f".{row['slide']}.tmp-",dir=root) as td:
  td=Path(td);cp=td/f"{row['slide']}_coordinates.npy";fp=td/f"{row['slide']}_tissue_fraction.npy";rp=td/f"{row['slide']}_spatial_v31_receipt.json"
  np.save(cp,coords,allow_pickle=False);np.save(fp,fractions,allow_pickle=False)
  receipt={k:v for k,v in result.items() if k!='tiles'};receipt.update({'schema_version':SCHEMA_VERSION,'algorithm':'Spatial-v3 canonical algorithm with v3.1 operational controls','canonical_parent_sha256':CANONICAL_PARENT_SHA256,'source_path':str(Path(row['path'])),'coordinate_sha256':sha256_file(cp),'tissue_fraction_sha256':sha256_file(fp),'thumbnail_reads':0,'wsi_hashes':0,'saved_tile_pixels':0,'embeddings':0,'validation_access':0,'protected_test_access':0,'camelyon17_access':0,'annotation_access':0,'training_operations':0,'evaluation_operations':0})
  rp.write_text(json.dumps(receipt,indent=2),encoding='utf-8');os.replace(td,final_dir)
 return json.loads((final_dir/f"{row['slide']}_spatial_v31_receipt.json").read_text(encoding='utf-8'))

def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 args=parse_args()
 if not args.execute_spatial_v31:raise SystemExit('REFUSAL: explicit --execute-spatial-v31 required')
 cfg=yaml.safe_load(project_path(args.config).read_text(encoding='utf-8-sig'));source=json.loads(project_path(cfg['processing_manifest']).read_text(encoding='utf-8-sig'));rows=source['slides']
 if len(rows)!=30 or sum(r['label']=='normal' for r in rows)!=15 or sum(r['label']=='tumor' for r in rows)!=15:raise RuntimeError('frozen 15+15 cohort gate failed')
 if args.slides:
  requested=set(args.slides);rows=[r for r in rows if r['slide'] in requested];missing=requested-{r['slide'] for r in rows}
  if missing:raise RuntimeError(f'Unavailable or prohibited: {sorted(missing)}')
 out=project_path(cfg['output_root']);slides_root=out/'slides';results=[]
 for i,row in enumerate(rows,1):
  if not args.force:
   prior=validate_restart(slides_root,row,int(cfg['max_tiles_per_slide']))
   if prior is not None:print(f"Reusing validated completed slide {row['slide']}");results.append(prior);continue
  print(f"Spatial v3.1 sampling {i}/{len(rows)}: {row['slide']} ({row['split']})")
  result,coords,fractions=sample_slide(row,cfg);results.append(atomic_write_slide(slides_root,row,result,coords,fractions))
 payload={'schema_version':SCHEMA_VERSION,'canonical_parent_sha256':CANONICAL_PARENT_SHA256,'slide_count':len(rows),'completed_count':len(results),'total_region_reads':sum(r['region_reads'] for r in results),'total_coordinates':sum(r['selected_tile_count'] for r in results),'thumbnail_reads':0,'wsi_hashes':0,'saved_tile_pixels':0,'embeddings':0,'validation_access':0,'protected_test_access':0,'evaluation_operations':0,'slides':results,'passed':len(results)==len(rows)}
 out.mkdir(parents=True,exist_ok=True);(out/'SHARD_RECEIPT.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');write_csv(out/'spatial_v31_sampling_summary.csv',[{k:v for k,v in r.items() if k!='validation'} for r in results]);print(json.dumps({k:v for k,v in payload.items() if k!='slides'},indent=2));print('PASS: Spatial-v3.1 operational-control sampler completed.')
if __name__=='__main__':main()
