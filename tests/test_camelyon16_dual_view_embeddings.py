import numpy as np
from core.wsi.materialize_camelyon16_dual_view_embeddings import TileDataset

def test_tile_dataset_length_without_opening_slide():
 dataset=TileDataset('missing.tif',1,np.asarray([[0,0],[512,512]]),256)
 assert len(dataset)==2
