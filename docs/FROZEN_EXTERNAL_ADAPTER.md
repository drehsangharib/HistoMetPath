# Frozen External Adapter

This milestone adapts the sealed CAMELYON17 cohort to the frozen HistoMetPath processing-manifest contract. The preflight opens only WSI metadata through OpenSlide to freeze dimensions, pyramid levels, MPP, vendor, and the 0.5-MPP level choice. It does not read tile pixels, generate sampler coordinates, create embeddings, predict, mutate the lock, or consume execution.

The original CAMELYON16 sampler command-line gates remain unchanged. The later activation runner will call the unchanged Spatial v2 and Spatial v3 `sample_slide()` functions with external manifest rows and frozen configuration values. Activation remains disabled until committed, CI-green, resealed, and re-preflighted.
