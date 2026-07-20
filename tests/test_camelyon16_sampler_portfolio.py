from analysis.audit_camelyon16_sampler_portfolio import jaccard,pareto_front

def test_jaccard():
 assert jaccard({(0,0),(1,1)},{(1,1),(2,2)})==1/3
 assert jaccard(set(),set())==1.0

def test_pareto_front():
 metrics={'a':{'x':1,'y':2},'b':{'x':2,'y':1},'c':{'x':1,'y':1}}
 assert set(pareto_front(metrics,['x','y']))=={'a','b'}
