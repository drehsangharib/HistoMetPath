"""Density-aware spatially distributed WSI sampler v2.

The sampler is annotation-independent and refuses test slides. It scans the
complete selected WSI level, guarantees one representative per occupied 10x10
bin, allocates the remaining 300-tile budget by candidate density using the
largest-remainder method, and performs deterministic farthest-point selection
within each bin.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import openslide
import yaml

from core.wsi.run_camelyon16_batch_pipeline import project_path
from core.wsi.run_camelyon16_spatial_sampler import spatial_bin
from core.wsi.tissue_mask import create_tissue_mask, tissue_fraction


def parse_args():
    p=argparse.ArgumentParser(description="Run development-only spatial sampler v2")
    p.add_argument("--config",default="configs/wsi/camelyon16_spatial_sampler_v2_development.yaml")
    p.add_argument("--slides",nargs="+",default=None)
    p.add_argument("--force",action="store_true")
    return p.parse_args()


def load_config(path: Path):
    cfg=yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    required={"processing_manifest","output_root","tile_size","stride","intensity_threshold","min_tissue_fraction","spatial_grid_rows","spatial_grid_columns","max_tiles_per_slide","minimum_tiles_per_occupied_bin","allowed_splits","prohibited_splits"}
    missing=sorted(required-set(cfg))
    if missing: raise KeyError(f"Missing config keys: {missing}")
    if "test" not in cfg["prohibited_splits"]: raise RuntimeError("Test split must be prohibited")
    return cfg


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


def farthest_point_select(candidates: list[dict], count: int) -> list[dict]:
    if count>=len(candidates):
        return sorted(candidates,key=lambda c:(-c["tissue_fraction"],c["y"],c["x"]))
    ordered=sorted(candidates,key=lambda c:(-c["tissue_fraction"],c["y"],c["x"]))
    selected=[ordered[0]]
    remaining=ordered[1:]
    while len(selected)<count:
        def score(candidate):
            minimum=min((candidate["x"]-item["x"])**2+(candidate["y"]-item["y"])**2 for item in selected)
            return (minimum,candidate["tissue_fraction"],-candidate["y"],-candidate["x"])
        chosen=max(remaining,key=score)
        selected.append(chosen); remaining.remove(chosen)
    return selected


def sample_slide(row,cfg):
    started=time.time(); slide=openslide.OpenSlide(str(Path(row["path"])))
    try:
        level=int(row["selected_level"]); downsample=float(row["selected_downsample"])
        level_width,level_height=slide.level_dimensions[level]; slide_width,slide_height=slide.dimensions
        tile_size=int(cfg["tile_size"]); stride=int(cfg["stride"]); rows=int(cfg["spatial_grid_rows"]); columns=int(cfg["spatial_grid_columns"])
        bins={}; examined=0
        for y_level in range(0,level_height-tile_size+1,stride):
            for x_level in range(0,level_width-tile_size+1,stride):
                examined+=1; x_zero=int(round(x_level*downsample)); y_zero=int(round(y_level*downsample))
                tile=slide.read_region((x_zero,y_zero),level,(tile_size,tile_size)).convert("RGB")
                fraction=tissue_fraction(create_tissue_mask(tile,intensity_threshold=int(cfg["intensity_threshold"])))
                if fraction<float(cfg["min_tissue_fraction"]): continue
                key=spatial_bin(x_zero,y_zero,slide_width,slide_height,rows,columns)
                bins.setdefault(key,[]).append({"x":x_zero,"y":y_zero,"tissue_fraction":float(fraction),"spatial_bin_row":key[0],"spatial_bin_column":key[1]})
    finally:
        slide.close()
    counts={key:len(value) for key,value in bins.items()}; target=min(int(cfg["max_tiles_per_slide"]),sum(counts.values()))
    allocation=allocate_density_budget(counts,target,int(cfg["minimum_tiles_per_occupied_bin"]))
    selected=[]
    for key in sorted(bins): selected.extend(farthest_point_select(bins[key],allocation[key]))
    selected.sort(key=lambda c:(c["spatial_bin_row"],c["spatial_bin_column"],c["y"],c["x"]))
    if len(selected)!=target: raise RuntimeError(f"{row['slide']}: selected {len(selected)} but expected {target}")
    coords=np.asarray([[c["x"],c["y"]] for c in selected],dtype=np.int64); fractions=np.asarray([c["tissue_fraction"] for c in selected],dtype=np.float32)
    return {"slide":row["slide"],"label":row["label"],"split":row["split"],"path":row["path"],"selected_level":level,"selected_downsample":downsample,"effective_mpp":row["effective_mpp"],"candidates_examined":examined,"tissue_candidate_count":sum(counts.values()),"occupied_bin_count":len(bins),"grid_bin_count":rows*columns,"selected_tile_count":len(selected),"budget_utilization_fraction":len(selected)/int(cfg["max_tiles_per_slide"]),"mean_tissue_fraction":float(fractions.mean()),"coordinate_x_min":int(coords[:,0].min()),"coordinate_x_max":int(coords[:,0].max()),"coordinate_y_min":int(coords[:,1].min()),"coordinate_y_max":int(coords[:,1].max()),"elapsed_seconds":float(time.time()-started),"bin_allocations":[{"row":key[0],"column":key[1],"candidate_count":counts[key],"allocated_count":allocation[key]} for key in sorted(counts)],"tiles":selected,"status":"complete"}


def write_csv(path,rows):
    flat=[{k:v for k,v in row.items() if k not in {"tiles","bin_allocations"}} for row in rows]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)


def main():
    args=parse_args(); cfg_path=project_path(args.config); cfg=load_config(cfg_path)
    source=json.loads(project_path(cfg["processing_manifest"]).read_text(encoding="utf-8")); allowed=set(cfg["allowed_splits"]); prohibited=set(cfg["prohibited_splits"])
    rows=[row for row in source["slides"] if row["split"] in allowed]
    if len(rows)!=36 or any(row["split"] in prohibited for row in rows): raise RuntimeError("Development-only split gate failed")
    if args.slides:
        requested=set(args.slides); rows=[row for row in rows if row["slide"] in requested]
        missing=sorted(requested-{row["slide"] for row in rows})
        if missing: raise RuntimeError(f"Requested slides unavailable or prohibited: {missing}")
    out=project_path(cfg["output_root"]); coord_root=out/"coordinates"; out.mkdir(parents=True,exist_ok=True); coord_root.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"spatial_v2_sampling_manifest.json"; prior={}
    if manifest_path.is_file() and not args.force:
        prior_data=json.loads(manifest_path.read_text(encoding="utf-8")); prior={row["slide"]:row for row in prior_data.get("slides",[])}
    results=[]
    for index,row in enumerate(rows,1):
        if row["slide"] in prior and prior[row["slide"]].get("status")=="complete":
            print(f"Skipping completed slide {row['slide']}"); results.append(prior[row["slide"]]); continue
        print(f"Spatial v2 sampling {index}/{len(rows)}: {row['slide']} ({row['split']})")
        result=sample_slide(row,cfg); results.append(result)
        np.save(coord_root/f"{row['slide']}_coordinates.npy",np.asarray([[t["x"],t["y"]] for t in result["tiles"]],dtype=np.int64),allow_pickle=False)
        np.save(coord_root/f"{row['slide']}_tissue_fractions.npy",np.asarray([t["tissue_fraction"] for t in result["tiles"]],dtype=np.float32),allow_pickle=False)
        payload={"schema_version":"2.0","dataset":cfg["dataset"],"scientific_scope":cfg["scientific_scope"],"development_slide_count":len(rows),"test_slides_loaded":0,"prohibited_splits":sorted(prohibited),"slides":results,"completed_count":len(results),"passed":len(results)==len(rows)}
        manifest_path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); write_csv(out/"spatial_v2_sampling_summary.csv",results)
    print(json.dumps({"development_slide_count":len(rows),"completed_count":len(results),"test_slides_loaded":0,"passed":len(results)==len(rows)},indent=2)); print("PASS: Density-aware development-only spatial sampler v2 completed.")

if __name__=="__main__": main()
