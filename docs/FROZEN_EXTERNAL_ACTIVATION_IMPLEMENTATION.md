# Execution-Disabled Frozen External Activation Implementation

This milestone freezes and tests the exact adapter-to-sampler-to-embedding-to-model callable contract while keeping pixel processing and consequential execution disabled. The preflight imports the unchanged Spatial v2/v3 `sample_slide` functions and frozen encoder/materializer functions but never calls them.

The activation plan is fixed at 20 external slides, 300 zero-level `(x,y)` coordinates per view, `(300,512)` embeddings per view, 512 primary mean-pooled features, 1024 secondary concatenated features, and the frozen primary threshold `0.2404209436418631`. No model fitting, threshold fitting, case exclusion, coordinate generation, embedding generation, prediction, lock mutation, or execution-count consumption occurs in this milestone.
