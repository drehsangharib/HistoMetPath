# Frozen External Runner Preflight

This preflight is intentionally non-consuming. It verifies the sealed external lock, checksum record, readiness report, frozen artifacts/configurations, repository commit, working-tree cleanliness, slide presence and sizes, and minimum free space. It does not open WSI pixel data, run sampling or embedding, load models for prediction, mutate the lock, or increment the one-time execution count.

A passing preflight authorizes construction and review of the exact one-time runner. It does not itself authorize ad hoc inference. The balanced, size-optimized CAMELYON17 cohort remains a constrained external pilot rather than a prevalence-representative validation cohort.
