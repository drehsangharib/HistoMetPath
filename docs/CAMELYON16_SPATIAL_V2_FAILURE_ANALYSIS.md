# CAMELYON16 Spatial Sampler v2 Failure Analysis

This post-selection diagnostic analyzes the seven development tumor slides still lacking annotation-intersecting tiles under Spatial Sampler v2.

The analysis rescans the frozen candidate grid and determines whether lesion-intersecting grid candidates existed, passed the tissue threshold, which bins contained them, how many selections those bins received, and the nearest selected-tile distance to lesion annotations.

Only training and validation slides are loaded. Test slides, embeddings, model probabilities, thresholds, and final-test outputs are excluded. The completed final test remains immutable.
