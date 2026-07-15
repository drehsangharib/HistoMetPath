from core.wsi.run_camelyon16_spatial_sampler import spatial_bin, select_from_bins


def test_spatial_bin_covers_full_extent():
    assert spatial_bin(0, 0, 1000, 2000, 10, 10) == (0, 0)
    assert spatial_bin(999, 1999, 1000, 2000, 10, 10) == (9, 9)


def test_selection_is_deterministic_and_distributed():
    candidates = {
        (0, 0): [(0.9, 10, 20), (0.8, 20, 20)],
        (9, 9): [(0.7, 900, 1800)],
    }
    first = select_from_bins(candidates, 300)
    second = select_from_bins(candidates, 300)
    assert first == second
    assert len(first) == 3
    assert {(row["spatial_bin_row"], row["spatial_bin_column"]) for row in first} == {(0, 0), (9, 9)}
