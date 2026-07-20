# CAMELYON16 Sampler Stability and Pareto Audit

This development-only audit compares raster sampling and Spatial Samplers v1-v3 across 36 training/validation slides and 18 annotated development tumor slides.

The audit quantifies lesion-positive bag coverage, total lesion-intersecting tiles, mean polygon coverage, spatial-bin occupancy, nearest-neighbor dispersion, and pairwise coordinate-set Jaccard overlap. It then identifies the nondominated sampler set under predeclared Pareto objectives.

No test slides, embeddings for modeling, probabilities, thresholds, or model outputs are loaded. The completed final-test lock is checksum-recorded and remains immutable.
