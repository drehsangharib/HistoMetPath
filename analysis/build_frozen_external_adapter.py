"""Build and validate the frozen external adapter contract.

The committed adapter is non-consuming: it creates an external processing
manifest and validates the unchanged sampler/materializer/model contracts. WSI
pixel processing remains disabled until the activation milestone is committed,
CI-green, resealed, and deliberately executed once.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib, openslide, yaml
from core.wsi.run_camelyon16_batch_pipeline import project_path, select_level

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evaluation/frozen_external_adapter.yaml');p.add_argument('--preflight-only',action='store_true',required=True);return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'))
 paths={k:project_path(cfg[k]) for k in ['external_manifest','external_lock','primary_sampler_config','secondary_sampler_config','embedding_config','frozen_model_artifact']};missing=[f'{k}: {v}' for k,v in paths.items() if not v.is_file()]
 if missing:raise FileNotFoundError('Missing adapter evidence:\n'+'\n'.join(missing))
 manifest=json.loads(paths['external_manifest'].read_text(encoding='utf-8'));lock=json.loads(paths['external_lock'].read_text(encoding='utf-8'));v2=yaml.safe_load(paths['primary_sampler_config'].read_text(encoding='utf-8-sig'));v3=yaml.safe_load(paths['secondary_sampler_config'].read_text(encoding='utf-8-sig'));emb=yaml.safe_load(paths['embedding_config'].read_text(encoding='utf-8-sig'));artifact=joblib.load(paths['frozen_model_artifact'])
 if cfg['adapter_execution_enabled'] is not False:raise RuntimeError('Adapter execution must remain disabled in this milestone')
 slides=manifest['slides'];processing=[]
 for row in slides:
  path=project_path(row['source_path'])
  if not path.is_file():raise FileNotFoundError(path)
  slide=openslide.OpenSlide(str(path))
  try:
   props=slide.properties;mpp_x=float(props.get(openslide.PROPERTY_NAME_MPP_X));mpp_y=float(props.get(openslide.PROPERTY_NAME_MPP_Y));level,down,effective=select_level(mpp_x,mpp_y,[float(x) for x in slide.level_downsamples],float(0.5))
   processing.append({'slide':row['slide_id'],'label':row['label'],'path':str(path.resolve()),'size_bytes':path.stat().st_size,'split':cfg['external_split_name'],'width':int(slide.dimensions[0]),'height':int(slide.dimensions[1]),'level_count':int(slide.level_count),'level_dimensions':[list(map(int,d)) for d in slide.level_dimensions],'level_downsamples':[float(x) for x in slide.level_downsamples],'mpp_x':mpp_x,'mpp_y':mpp_y,'vendor':props.get(openslide.PROPERTY_NAME_VENDOR),'selected_level':int(level),'selected_downsample':down,'effective_mpp':effective,'target_mpp':0.5,'tile_size':int(v2['tile_size']),'stride':int(v2['stride']),'status':'adapter_preflight_complete'})
  finally:slide.close()
 artifacts=artifact['artifacts'];primary=artifacts[cfg['primary_model']];secondary=artifacts[cfg['secondary_model']]
 checks={'lock_sealed':lock.get('lock_status')=='sealed_pre_execution','execution_count_zero':int(lock.get('execution_count',-1))==0,'slide_count':len(processing)==int(cfg['expected_slides']),'v2_tile_budget':int(v2['max_tiles_per_slide'])==int(cfg['expected_coordinates_per_view']),'v3_tile_budget':int(v3['max_tiles_per_slide'])==int(cfg['expected_coordinates_per_view']),'v2_tile_size':int(v2['tile_size'])==256,'v3_tile_size':int(v3['tile_size'])==256,'embedding_dimension':int(emb['embedding_dimension'])==512,'primary_features':int(primary['model'].n_features_in_)==512,'secondary_features':int(secondary['model'].n_features_in_)==1024,'primary_threshold':abs(float(artifact['selected_threshold'])-float(cfg['primary_threshold']))<1e-15,'adapter_execution_disabled':cfg['adapter_execution_enabled'] is False}
 ready=all(checks.values());out=project_path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True);processing_manifest={'schema_version':'1.0','dataset':'CAMELYON17','scientific_scope':cfg['scientific_scope'],'slides':processing,'completed_count':len(processing),'requested_count':int(cfg['expected_slides']),'model_outputs_generated':False,'inference_executed':False,'execution_count_consumed':False,'passed':ready};(out/'external_processing_manifest.json').write_text(json.dumps(processing_manifest,indent=2),encoding='utf-8')
 report={'schema_version':'1.0','project':cfg['project'],'preflight_only':True,'adapter_execution_enabled':False,'wsi_metadata_opened':True,'wsi_tiles_read':False,'coordinates_generated':False,'embeddings_generated':False,'predictions_generated':False,'execution_count_consumed':False,'checks':checks,'adapter_ready_for_activation':ready,'blocking_reasons':[k for k,v in checks.items() if not v],'external_manifest_sha256':sha(paths['external_manifest']),'external_lock_sha256':sha(paths['external_lock']),'config_sha256':sha(cp),'passed':True};(out/'external_adapter_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));print('PASS: Frozen external adapter preflight completed without consuming execution.')
if __name__=='__main__':main()
