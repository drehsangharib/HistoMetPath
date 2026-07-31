# CAMELYON16 Training-Only Slide-Level Uncertainty and Calibration

This audit removes repeated-row pseudoreplication by aggregating ten repeated out-of-fold probabilities to one mean probability per training slide and model. The 30 training slides are the independent biological units.

For Spatial v2 and dual-view concatenation, the audit reports slide-level Brier score, expected calibration error, AUROC, AUPRC, accuracy, balanced accuracy, repeated probability standard deviation and quantiles, persistent correctness/incorrectness, split sensitivity, and cross-model slide disagreement.

No model is refit. Validation and test slides or predictions are not loaded or generated. The frozen validation result and completed final test are checksum-recorded as immutable external evidence.
