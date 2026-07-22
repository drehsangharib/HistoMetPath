"""Materialize frozen-encoder embeddings for Spatial v2 and v3 development views."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np, openslide, torch, yaml
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
from core.wsi.run_camelyon16_batch_pipeline import project_path

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_dual_view_embedding_development.yaml');p.add_argument('--slides',nargs='+');p.add_argument('--force',action='store_true');return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def build_encoder(cfg,device):
 if cfg['backbone']!='resnet18':raise RuntimeError('Only resnet18 is frozen for this milestone')
 model=models.resnet18(weights=None);model.fc=torch.nn.Identity();checkpoint=torch.load(project_path(cfg['checkpoint']),map_location='cpu');state=checkpoint.get('state_dict',checkpoint);target=model.state_dict();mapped={}
 prefixes=('model.backbone.','backbone.','encoder.','model.encoder.','model.')
 for key,value in state.items():
  candidates=[key]+[key[len(p):] for p in prefixes if key.startswith(p)]
  for candidate in candidates:
   if candidate in target and target[candidate].shape==value.shape: mapped[candidate]=value;break
 missing=[k for k in target if k not in mapped and not k.startswith('fc.')]
 coverage=len(mapped)/max(1,len([k for k in target if not k.startswith('fc.')]))
 if coverage<0.90:raise RuntimeError(f'Checkpoint/backbone coverage too low: {coverage:.3f}; missing={missing[:10]}')
 target.update(mapped);model.load_state_dict(target);model.to(device).eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 return model,coverage
class TileDataset(Dataset):
 def __init__(self,slide_path,level,coordinates,tile_size):
  self.slide_path=str(slide_path);self.level=int(level);self.coordinates=np.asarray(coordinates,dtype=np.int64);self.tile_size=int(tile_size);self.slide=None;self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
 def __len__(self):return len(self.coordinates)
 def __getitem__(self,index):
  if self.slide is None:self.slide=openslide.OpenSlide(self.slide_path)
  x,y=map(int,self.coordinates[index]);image=self.slide.read_region((x,y),self.level,(self.tile_size,self.tile_size)).convert('RGB');return self.transform(image)
def embed(model,row,coords,cfg,device):
 data=TileDataset(Path(row['path']),row['selected_level'],coords,cfg['tile_size']);loader=DataLoader(data,batch_size=int(cfg['batch_size']),shuffle=False,num_workers=int(cfg['num_workers']),pin_memory=device.type=='cuda');chunks=[]
 with torch.inference_mode():
  for batch in loader:chunks.append(model(batch.to(device,non_blocking=True)).cpu().numpy().astype(np.float32))
 output=np.concatenate(chunks,axis=0)
 if output.shape!=(len(coords),int(cfg['embedding_dimension'])):raise RuntimeError(f"{row['slide']}: embedding shape {output.shape}")
 if not np.isfinite(output).all():raise RuntimeError(f"{row['slide']}: nonfinite embeddings")
 return output
def main():
 args=parse_args();cp=project_path(args.config);cfg=yaml.safe_load(cp.read_text(encoding='utf-8-sig'));source_path=project_path(cfg['processing_manifest']);source=json.loads(source_path.read_text(encoding='utf-8'));allowed=set(cfg['allowed_splits']);prohibited=set(cfg['prohibited_splits']);rows=[r for r in source['slides'] if r['split'] in allowed]
 if len(rows)!=36 or any(r['split'] in prohibited for r in rows):raise RuntimeError('Development-only split gate failed')
 if args.slides:
  requested=set(args.slides);rows=[r for r in rows if r['slide'] in requested];missing=requested-{r['slide'] for r in rows}
  if missing:raise RuntimeError(f'Unavailable or prohibited slides: {sorted(missing)}')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,coverage=build_encoder(cfg,device);out=project_path(cfg['output_root']);mr=project_path(cfg['manifest_root']);out.mkdir(parents=True,exist_ok=True);mr.mkdir(parents=True,exist_ok=True);records=[]
 roots={'spatial_v2':project_path(cfg['v2_coordinate_root']),'spatial_v3':project_path(cfg['v3_coordinate_root'])}
 for row in sorted(rows,key=lambda r:r['slide']):
  for view in cfg['views']:
   coordinate_path=roots[view]/f"{row['slide']}_coordinates.npy";coordinates=np.load(coordinate_path,allow_pickle=False);view_root=out/view;view_root.mkdir(parents=True,exist_ok=True);embedding_path=view_root/f"{row['slide']}_embeddings.npy";coordinate_copy=view_root/f"{row['slide']}_coordinates.npy"
   if embedding_path.is_file() and coordinate_copy.is_file() and not args.force:
    embeddings=np.load(embedding_path,allow_pickle=False)
    if embeddings.shape==(len(coordinates),int(cfg['embedding_dimension'])):print(f"Skipping completed {view}: {row['slide']}");records.append({'slide':row['slide'],'label':row['label'],'split':row['split'],'view':view,'tile_count':len(coordinates),'embedding_dimension':embeddings.shape[1],'coordinate_sha256':sha(coordinate_path),'embedding_sha256':sha(embedding_path),'status':'complete'});continue
   print(f"Embedding {view}: {row['slide']} ({row['split']})");started=time.time();embeddings=embed(model,row,coordinates,cfg,device);np.save(embedding_path,embeddings,allow_pickle=False);np.save(coordinate_copy,coordinates.astype(np.int64),allow_pickle=False);records.append({'slide':row['slide'],'label':row['label'],'split':row['split'],'view':view,'tile_count':len(coordinates),'embedding_dimension':embeddings.shape[1],'coordinate_sha256':sha(coordinate_path),'embedding_sha256':sha(embedding_path),'elapsed_seconds':time.time()-started,'status':'complete'})
 payload={'schema_version':'1.0','dataset':cfg['dataset'],'scientific_scope':cfg['scientific_scope'],'development_slide_count':len(rows),'view_count':len(cfg['views']),'record_count':len(records),'test_slides_loaded':0,'model_outputs_generated':False,'device':str(device),'checkpoint_sha256':sha(project_path(cfg['checkpoint'])),'encoder_state_coverage':coverage,'processing_manifest_sha256':sha(source_path),'records':records,'passed':len(records)==len(rows)*len(cfg['views'])};manifest=mr/'dual_view_embedding_manifest.json';manifest.write_text(json.dumps(payload,indent=2),encoding='utf-8')
 with (mr/'dual_view_embedding_manifest.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2));print('PASS: Development-only dual-view embeddings materialized.')
if __name__=='__main__':main()
