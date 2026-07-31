from analysis.audit_external_cohort_readiness import normalize

def test_identifier_normalization():
 assert normalize(' Tumor_001 ')=='tumor_001'
