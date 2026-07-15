# CAMELYON16 Annotation-Aware Lesion-Coverage Audit

This post-test diagnostic compares the locked sampled tile coordinates with CAMELYON16 XML lesion polygons for all 21 tumor slides in the 42-slide cohort.

The audit determines whether each slide bag contained at least one annotation-intersecting tile, how many sampled tiles intersected lesions, and how many annotated polygons were represented.

## Scientific boundary

The one-time fresh-test result is complete and immutable. This audit does not generate model probabilities, change thresholds, retrain models, or revise the final test. Any future sampling improvement requires a new development protocol and an entirely new untouched test boundary.
