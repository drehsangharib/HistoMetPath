# HistoMetPath Development Model Card

## Intended status
Research-development artifact only. Not clinically validated and not intended for diagnosis or patient management.

## Frozen primary baseline
Spatial v2 coordinate sampling, frozen ResNet-18 embeddings, slide-level mean pooling, and logistic regression.

## Secondary exploratory candidate
Dual-view mean concatenation using Spatial v2 and Spatial v3 embeddings.

## Evidence boundary
The historical held-out CAMELYON16 test was executed once and is immutable. It is a small development benchmark and cannot evaluate later sampler or model revisions. Any new comparative performance claim requires a new untouched cohort.

## Limitations
Small sample sizes, unstable slide-level predictions, poor calibration, weak selective-prediction behavior, no external prospective validation, and no clinical-use claim.
