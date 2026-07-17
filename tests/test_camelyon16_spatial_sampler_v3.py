import numpy as np
from PIL import Image
from core.wsi.run_camelyon16_spatial_sampler_v3 import descriptor,joint_farthest_select

def test_descriptor_is_finite_and_fixed_length():
 value=descriptor(Image.new('RGB',(16,16),(120,80,160)))
 assert value.shape==(12,)
 assert np.isfinite(value).all()

def test_joint_selection_is_deterministic():
 rows=[{'x':0,'y':0,'tissue_fraction':.9,'descriptor':[0.]*12},{'x':100,'y':0,'tissue_fraction':.8,'descriptor':[1.]*12},{'x':50,'y':0,'tissue_fraction':.95,'descriptor':[.5]*12}]
 assert joint_farthest_select(rows,2)==joint_farthest_select(rows,2)
