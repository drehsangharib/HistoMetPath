from analysis.diagnose_camelyon16_spatial_v2_failures import categorize


def test_failure_categories_are_deterministic():
    assert categorize({
        "lesion_grid_candidate_count": 0,
        "lesion_tissue_eligible_candidate_count": 0,
        "lesion_bin_selected_tile_count": 0,
    }) == "grid_resolution_or_lesion_geometry_miss"
    assert categorize({
        "lesion_grid_candidate_count": 2,
        "lesion_tissue_eligible_candidate_count": 0,
        "lesion_bin_selected_tile_count": 0,
    }) == "tissue_threshold_exclusion"
    assert categorize({
        "lesion_grid_candidate_count": 2,
        "lesion_tissue_eligible_candidate_count": 1,
        "lesion_bin_selected_tile_count": 0,
    }) == "allocation_exclusion"
    assert categorize({
        "lesion_grid_candidate_count": 2,
        "lesion_tissue_eligible_candidate_count": 1,
        "lesion_bin_selected_tile_count": 3,
    }) == "within_bin_selection_miss"
