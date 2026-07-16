"""Compare raster, spatial v1, and spatial v2 lesion coverage on development tumors."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import yaml
from analysis.audit_camelyon16_lesion_coverage import audit_slide
from core.wsi.run_camelyon16_batch_pipeline import PROJECT_ROOT,project_path

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/wsi/camelyon16_spatial_sampler_v2_development.yaml"); return p.parse_args()
def main():
    cfg=yaml.safe_load(project_path(parse_args().config).read_text(encoding="utf-8-sig")); source=json.loads(project_path(cfg["processing_manifest"]).read_text(encoding="utf-8"))
    roots={"raster":project_path(cfg["raster_coordinate_root"]),"spatial_v1":project_path(cfg["spatial_v1_coordinate_root"]),"spatial_v2":project_path(cfg["output_root"])/"coordinates"}; annotations=project_path(cfg["annotation_root"])
    rows=sorted([r for r in source["slides"] if r["label"]=="tumor" and r["split"] in {"train","validation"}],key=lambda r:r["slide"])
    if len(rows)!=18: raise RuntimeError("Expected 18 development tumor slides")
    output=[]
    for row in rows:
        audits={name:audit_slide(row,root/f"{row['slide']}_coordinates.npy",annotations/f"{row['slide']}.xml",int(cfg["tile_size"])) for name,root in roots.items()}
        output.append({"slide":row["slide"],"split":row["split"],**{f"{name}_has_lesion":audit["bag_contains_annotated_lesion"] for name,audit in audits.items()},**{f"{name}_lesion_tiles":audit["lesion_intersecting_tile_count"] for name,audit in audits.items()},**{f"{name}_polygon_fraction":audit["covered_polygon_fraction"] for name,audit in audits.items()}})
    summary={"development_tumor_slides":18,"test_slides_loaded":0,"raster_bags_with_lesion":sum(r["raster_has_lesion"] for r in output),"spatial_v1_bags_with_lesion":sum(r["spatial_v1_has_lesion"] for r in output),"spatial_v2_bags_with_lesion":sum(r["spatial_v2_has_lesion"] for r in output),"v2_improvement_over_v1":sum(r["spatial_v2_has_lesion"] for r in output)-sum(r["spatial_v1_has_lesion"] for r in output),"slides":output,"passed":True}
    out=project_path(cfg["output_root"]); (out/"spatial_sampler_comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"spatial_sampler_comparison.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    print(json.dumps({k:v for k,v in summary.items() if k!="slides"},indent=2)); print("PASS: Development-only sampler comparison completed.")
if __name__=="__main__": main()
