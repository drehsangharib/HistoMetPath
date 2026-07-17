# CAMELYON16 Spatial Sampler v3 Development

Spatial Sampler v3 preserves v2 density-aware bin allocation but replaces pure spatial farthest-point selection with deterministic joint diversity in spatial and morphology-descriptor space. Descriptors include RGB moments, saturation moments, edge density, and optical-density moments. Annotations are never used for tile selection.

Only 30 training and six validation slides are eligible. Test slides and model outputs are prohibited. Lesion annotations are used only after the frozen sampler run to compare raster and Spatial Samplers v1-v3. The completed final test remains immutable.
