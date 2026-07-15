# CAMELYON16 Spatial Sampler Development

This milestone replaces raster-order truncation with annotation-independent spatial sampling for training and validation slides only.

The complete WSI candidate grid is scanned at the locked physical resolution. Tissue-rich candidates are assigned to a fixed 10x10 spatial grid, and up to three candidates with the highest tissue fraction are selected per occupied bin, for a maximum of 300 tiles per slide.

Annotations are never used by the sampler. They are used only after sampling to compare lesion coverage on 18 training/validation tumor slides. Test slides are explicitly prohibited and `test_slides_loaded` must remain zero.

The completed final-test result remains immutable. Any future model-performance evaluation requires a new untouched test boundary.
