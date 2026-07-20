from core.wsi.build_camelyon16_v2_v3_consensus_sampler import build_consensus

def test_consensus_preserves_shared_and_balances_unique():
 v2=[(0,0),(1,1),(2,2),(3,3)];v3=[(0,0),(1,1),(4,4),(5,5)]
 selected,stats=build_consensus(v2,v3,6)
 assert set([(0,0),(1,1)]).issubset(selected)
 assert len(selected)==6
 assert stats['v2_unique_selected']==2
 assert stats['v3_unique_selected']==2

def test_consensus_is_deterministic():
 a=[(i,0) for i in range(10)];b=[(i,1) for i in range(10)]
 assert build_consensus(a,b,10)==build_consensus(a,b,10)
