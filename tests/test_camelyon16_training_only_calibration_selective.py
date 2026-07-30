import numpy as np
from analysis.audit_camelyon16_training_only_calibration_selective import ece,selective_rows

def test_ece_perfect_predictions():
 y=np.asarray([0,0,1,1]);p=np.asarray([0.,0.,1.,1.]);score,rows=ece(y,p,2);assert score==0.;assert sum(r['count'] for r in rows)==4

def test_selective_curve_retains_requested_rows():
 y=np.asarray([0,0,1,1]);p=np.asarray([.1,.4,.6,.9]);slides=np.asarray(['a','b','c','d']);rows=selective_rows(y,p,slides,[1.,.5],.5);assert rows[0]['retained_count']==4;assert rows[1]['retained_count']==2
