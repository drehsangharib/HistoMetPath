# CAMELYON16 Training-Only Calibration and Selective Prediction

This diagnostic evaluates calibration and selective prediction for the frozen Spatial v2 primary baseline and dual-view concatenation secondary candidate using existing repeated out-of-fold predictions from 30 training slides.

It reports Brier score, five-bin expected calibration error, reliability-bin data, confidence-ranked coverage-versus-accuracy curves, and an agreement-only policy. No model is refit. Validation and test embeddings or probabilities are not loaded or generated. The six-slide validation result and completed final test are checksum-recorded as immutable external evidence.
