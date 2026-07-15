"""Compare raster and spatial lesion coverage on train/validation tumors only."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, yaml
from analysis.audit_camelyon16_lesion_coverage import audit_slide
from core.wsi.run_camelyon16_batch_pipeline import project_path


def parse_args():
    p=argparse.ArgumentParser(description="Compare development-only sampling coverage")
    p.add_argument("--config",default="configs/wsi/camelyon16_spatial_sampler_development.yaml")
    return p.parse_args()

def main():
    args=parse_args(); config_path=project_path(args.config); cfg=yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    source=json.loads(project_path(cfg["processing_manifest"]).read_text(encoding="utf-8"))
    spatial_root=project_path(cfg["output_root"])/"coordinates"; raster_root=PROJECT_ROOT/"embeddings"/"camelyon16"/"expanded_fresh_holdout"; annotation_root=project_path(cfg["annotation_root"])
    rows=[r for r in source["slides"] if r["label"]=="tumor" and r["split"] in {"train","validation"}]
    if len(rows)!=18: raise RuntimeError(f"Expected 18 development tumor slides; found {len(rows)}")
    output=[]
    for row in sorted(rows,key=lambda item:item["slide"]):
        xml=annotation_root/f"{row['slide']}.xml"
        raster=audit_slide(row,raster_root/f"{row['slide']}_coordinates.npy",xml,int(cfg["tile_size"]))
        spatial=audit_slide(row,spatial_root/f"{row['slide']}_coordinates.npy",xml,int(cfg["tile_size"]))
        output.append({"slide":row["slide"],"split":row["split"],"raster_has_lesion":raster["bag_contains_annotated_lesion"],"spatial_has_lesion":spatial["bag_contains_annotated_lesion"],"raster_lesion_tiles":raster["lesion_intersecting_tile_count"],"spatial_lesion_tiles":spatial["lesion_intersecting_tile_count"],"raster_polygon_fraction":raster["covered_polygon_fraction"],"spatial_polygon_fraction":spatial["covered_polygon_fraction"]})
    summary={"development_tumor_slides":18,"test_slides_loaded":0,"raster_bags_with_lesion":sum(r["raster_has_lesion"] for r in output),"spatial_bags_with_lesion":sum(r["spatial_has_lesion"] for r in output),"spatial_improvement_count":sum((not r["raster_has_lesion"]) and r["spatial_has_lesion"] for r in output),"slides":output,"passed":True}
    root=project_path(cfg["output_root"]); (root/"sampling_coverage_comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (root/"sampling_coverage_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    print(json.dumps({k:v for k,v in summary.items() if k!="slides"},indent=2)); print("PASS: Development-only raster-versus-spatial coverage comparison completed.")

if __name__=="__main__":
    from core.wsi.run_camelyon16_batch_pipeline import PROJECT_ROOT
    main()
