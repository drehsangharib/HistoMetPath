# Frozen CAMELYON17 External Pilot Protocol

This protocol seals a one-time, frozen external pilot before inference. The cohort contains 20 independent patients, 10 normal and 10 tumor slides, with two slides per class from each of five CAMELYON17 centers. One slide is used per patient.

## Selection limitation
The pilot is label-balanced and size-optimized for storage feasibility. It is not a random or prevalence-representative sample. Accuracy, balanced accuracy, AUROC, and AUPRC may be reported as constrained pilot results; prevalence-sensitive interpretation and definitive external-validation claims are prohibited.

## Frozen components
The Spatial v2 primary model, dual-view concatenation secondary model, encoder checkpoint, sampler configurations, preprocessing, model artifacts, and decision rules are checksum-sealed before inference. Model fitting, threshold fitting, sampler changes, encoder changes, and post-inference case exclusion are prohibited.

## Execution rule
Full WSI SHA-256 receipts are required before execution authorization. The evaluation may run once. Results must be sealed immediately afterward. The historical CAMELYON16 final test remains immutable and unrelated to this external execution count.
