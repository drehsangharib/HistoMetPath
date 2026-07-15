import numpy as np

from analysis.audit_camelyon16_lesion_coverage import (
    point_in_polygon,
    rectangle_intersects_polygon,
)


def test_rectangle_polygon_intersection():
    polygon = np.array(
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]
    )
    assert rectangle_intersects_polygon(15.0, 15.0, 25.0, 25.0, polygon)
    assert not rectangle_intersects_polygon(30.0, 30.0, 40.0, 40.0, polygon)
    assert point_in_polygon(12.0, 12.0, polygon)
    assert not point_in_polygon(25.0, 25.0, polygon)
