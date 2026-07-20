"""Compare v2, v3, and consensus lesion coverage on development tumors only."""
import argparse,csv,json
import yaml
from analysis.audit_camelyon16_lesion_coverage import audit_slide
from core.wsi.run_camelyon16_batch_pipeline import project_path
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/wsi/camelyon16_v2_v3_consensus_development.yaml');cfg=yaml.safe_load(project_path(p.parse_args().config).read_text(encoding='utf-8-sig'));m=json.loads(project_path(cfg['processing_manifest']).read_text(encoding='utf-8'));roots={'v2':project_path(cfg['v2_coordinate_root']),'v3':project_path(cfg['v3_coordinate_root']),'consensus':project_path(cfg['output_root'])/'coordinates'};ar=project_path(cfg['annotation_root']);rows=sorted([r for r in m['slides'] if r['label']=='tumor' and r['split'] in {'train','validation'}],key=lambda r:r['slide']);outrows=[]
 for row in rows:
  audits={n:audit_slide(row,root/f"{row['slide']}_coordinates.npy",ar/f"{row['slide']}.xml",256) for n,root in roots.items()};outrows.append({'slide':row['slide'],'split':row['split'],**{f'{n}_has_lesion':a['bag_contains_annotated_lesion'] for n,a in audits.items()},**{f'{n}_lesion_tiles':a['lesion_intersecting_tile_count'] for n,a in audits.items()},**{f'{n}_polygon_fraction':a['covered_polygon_fraction'] for n,a in audits.items()}})
 summary={'development_tumor_slides':18,'test_slides_loaded':0,'model_outputs_generated':False,**{f'{n}_bags_with_lesion':sum(r[f"{n}_has_lesion"] for r in outrows) for n in roots},**{f'{n}_total_lesion_tiles':sum(r[f"{n}_lesion_tiles"] for r in outrows) for n in roots},**{f'{n}_mean_polygon_coverage':sum(r[f"{n}_polygon_fraction"] for r in outrows)/18 for n in roots},'slides':outrows,'passed':True};out=project_path(cfg['output_root']);(out/'consensus_comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 with (out/'consensus_comparison.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
 print(json.dumps({k:v for k,v in summary.items() if k!='slides'},indent=2));print('PASS: Development-only consensus coverage comparison completed.')
if __name__=='__main__':main()
