# CAMELYON16 Spatial v2 versus Concatenation Disagreement Audit

This development-only diagnostic compares the primary frozen Spatial v2 mean-pooling logistic-regression baseline with the secondary dual-view mean-concatenation candidate using the existing repeated out-of-fold predictions from 30 training slides.

The audit does not refit models and does not load validation or test embeddings. It measures paired probability correlation, absolute probability differences, binary prediction disagreement at the frozen 0.5 training-only audit threshold, slide-level disagreement frequency, repeated prediction variability, uncertainty-band membership, and comparative correctness counts.

The existing six-slide validation result remains frozen and is checksum-recorded as external development evidence. The completed final test remains immutable, and no validation or test outputs are generated.
