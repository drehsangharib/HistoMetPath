# CAMELYON16 Dual-View Development Baselines

This milestone compares five interpretable training/validation-only baselines using frozen mean-pooled Spatial v2 and Spatial v3 embeddings: v2-only logistic regression, v3-only logistic regression, concatenated mean representations, late probability averaging, and leakage-safe logistic stacking trained from five-fold out-of-fold training probabilities.

All model fitting uses 30 training slides. Threshold selection and model comparison use six validation slides. No test slide or test prediction is loaded or generated. The selected result is a development decision only and cannot be evaluated on the consumed final-test cohort.
