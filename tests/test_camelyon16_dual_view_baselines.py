import numpy as np
from analysis.run_camelyon16_dual_view_baselines import select_threshold,metric_block

def test_threshold_selection_is_deterministic():
 y=np.asarray([0,0,1,1]);p=np.asarray([.1,.2,.7,.8]);assert select_threshold(y,p)==select_threshold(y,p)
def test_metric_schema():
 metrics=metric_block(np.asarray([0,1]),np.asarray([.1,.9]),.5);assert metrics['balanced_accuracy']==1.0;assert metrics['confusion_matrix']==[[1,0],[0,1]]
