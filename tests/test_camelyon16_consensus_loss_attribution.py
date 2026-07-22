import numpy as np

from analysis.audit_camelyon16_consensus_losses import (
    covered_polygon_fraction,
    lesion_coordinates,
)


def test_lesion_coordinate_attribution():
    polygon = np.asarray(
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]
    )
    coordinates = {(0, 0), (100, 100)}
    hits = lesion_coordinates(coordinates, [polygon], footprint=16.0)
    assert hits == {(0, 0)}
    assert covered_polygon_fraction(coordinates, [polygon], footprint=16.0) == 1.0
