# HistoMetPath Development Decision

This milestone freezes the current internal CAMELYON16 development conclusion. Spatial v2 mean-pooling logistic regression remains the primary development baseline because it tied dual-view concatenation on the frozen six-slide validation check and is simpler. Dual-view concatenation remains a secondary exploratory candidate because it ranked first in training-only repeated stability and had higher slide-level AUROC/AUPRC, but its advantage was modest and heterogeneous.

OOF stacking, equal-quota consensus compression as a primary sampler, confidence-based abstention, and agreement-only abstention are discontinued. Raw probabilities are not calibrated tumor-risk estimates. The completed final test is immutable and cannot evaluate subsequent development. A new untouched cohort is required before any new performance claim.
