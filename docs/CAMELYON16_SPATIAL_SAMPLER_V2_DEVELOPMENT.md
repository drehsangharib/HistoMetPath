# CAMELYON16 Spatial Sampler v2 Development

Spatial Sampler v2 remains annotation-independent and development-only. It scans all tissue candidates, guarantees one representative per occupied 10x10 bin, allocates the remaining 300-tile budget by candidate density with deterministic largest-remainder allocation, and uses deterministic farthest-point selection within each bin.

The sampler loads only 30 training and six validation slides. XML annotations are used only after selection to compare lesion coverage across raster, Spatial v1, and Spatial v2 on 18 development tumor slides. The completed final-test slides and results remain immutable and excluded.
