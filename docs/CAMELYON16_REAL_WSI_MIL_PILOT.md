# CAMELYON16 Real-WSI MIL Development Pilot

This milestone compares mean pooling, max pooling, and Attention MIL using 22 genuine CAMELYON16 slide bags.

## Leakage-safe protocol

- Training: 14 slides (7 normal, 7 tumor)
- Validation: 4 slides (2 normal, 2 tumor)
- Test: 4 held-out slides (2 normal, 2 tumor)
- Logistic-regression scaling is fitted on training slides only.
- Attention feature standardization is fitted on tile instances from training slides only.
- Attention early stopping uses validation loss only.
- The Attention model is a predeclared five-seed probability ensemble.
- Decision thresholds are selected on validation only and then frozen for test.
- Individual validation and test slide predictions are exported.

## Critical limitations

The four-slide test set is extremely small. Aggregate AUROC, AUPRC, accuracy, and balanced accuracy can change dramatically after a single prediction. Results must be reported with individual slide predictions and must not be presented as a clinical-performance estimate.

The patch encoder was originally trained on PCAM patches. The deterministic 300-tile cap may undersample small metastases. A larger cohort, annotation-aware sampling study, and external validation are required before stable real-WSI performance claims.
