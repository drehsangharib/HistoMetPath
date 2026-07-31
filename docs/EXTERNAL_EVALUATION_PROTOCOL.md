# HistoMetPath External Evaluation Protocol

A new comparative performance claim requires a genuinely untouched cohort. Before any model inference, the candidate cohort must pass a readiness audit confirming complete metadata, class minimums, unique slide and patient identifiers, accessible source files, and zero overlap with the existing CAMELYON16 development/test inventory.

The primary Spatial v2 model, secondary concatenation model, encoder, sampler, thresholds, and preprocessing must remain frozen. No cohort labels may be used to alter models, thresholds, sampling, encoders, preprocessing, or exclusion rules. Evaluation results must be generated once under a locked manifest and preserved with checksums.

The default minimum is 20 independent slides with at least 10 per class. A larger, multi-site cohort is strongly preferred.
