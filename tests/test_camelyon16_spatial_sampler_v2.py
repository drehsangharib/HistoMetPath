from core.wsi.run_camelyon16_spatial_sampler_v2 import allocate_density_budget, farthest_point_select

def test_density_budget_uses_full_capacity():
    counts={(0,0):100,(0,1):50,(1,0):25}
    allocation=allocate_density_budget(counts,120,1)
    assert sum(allocation.values())==120
    assert all(allocation[key]>=1 for key in counts)
    assert allocation[(0,0)]>allocation[(0,1)]>allocation[(1,0)]

def test_farthest_point_selection_is_deterministic():
    candidates=[{"x":0,"y":0,"tissue_fraction":0.9},{"x":100,"y":0,"tissue_fraction":0.8},{"x":50,"y":0,"tissue_fraction":0.95}]
    assert farthest_point_select(candidates,2)==farthest_point_select(candidates,2)
    selected=farthest_point_select(candidates,2)
    assert {item["x"] for item in selected}=={0,50}
