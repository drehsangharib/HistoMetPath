import numpy as np
from analysis.audit_camelyon16_training_only_slide_level_calibration import ece,quantile

def test_ece_uses_independent_records():
 y=np.asarray([0,1]);p=np.asarray([0.1,0.9]);score,rows=ece(y,p,2);assert abs(score-0.1)<1e-12;assert sum(r['count'] for r in rows)==2

def test_quantile():
 assert quantile([0.,1.],.5)==.5
