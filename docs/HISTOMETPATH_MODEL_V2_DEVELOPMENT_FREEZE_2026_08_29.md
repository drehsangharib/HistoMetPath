# HistoMetPath Model-v2 Development Freeze

**Freeze date:** 2026-08-29  
**Repository baseline:** `bc7e1af3186d5776e1c8e84d1c070a34a3e7993d`  
**Evidence archive root:** `D:\HistoMetPath-Evidence-Freeze\2026-08-29`  

## Frozen decision

- Retain `v3_300` as the slide-classification baseline.
- Retain Hybrid Spatial v4.3 as the sampler-development champion only.
- Retain the frozen PCAM ResNet-18 encoder as a useful feature extractor.
- Reject larger global bags, component-aware summaries, weakly supervised tile ranking, and the tested annotation-supervised lesion-evidence pooling method for classification promotion.
- Keep all six development-validation embedding arrays unloaded.
- Do not rerun the immutable CAMELYON16 final test.
- Keep CAMELYON17 sealed and unconsumed.

## Frozen training-only classification evidence

| Method | AUROC | AUPRC | Balanced accuracy | Sensitivity | Specificity | Decision |
|---|---:|---:|---:|---:|---:|---|
| Fixed `v3_300` | 0.6818 | 0.7102 | 0.6483 | 0.6733 | 0.6233 | Baseline |
| Fixed `v41_600` | 0.6691 | 0.7144 | 0.6183 | 0.6133 | 0.6233 | Reject |
| Fixed `v43_full` | 0.6578 | 0.6995 | 0.6167 | 0.5933 | 0.6400 | Reject for classification |
| Component-aware aggregation | 0.6427 | 0.6806 | 0.6183 | 0.6500 | 0.5867 | Reject |
| Fold-local weak tile ranking | 0.6187 | 0.6702 | 0.5983 | 0.5767 | 0.6200 | Reject |
| Annotation-supervised pooling attempt | 0.5910 | 0.6356 | 0.5783 | 0.6533 | 0.5033 | Reject |

All values are development estimates from 20 repeated five-fold outer evaluations over 30 balanced training slides. They are not independent-cohort performance claims.

## Sampler and encoder diagnostics

- Hybrid Spatial v4.3 passed sampler-development lesion-coverage gates with 17 of 18 lesion-positive development bags and 533 lesion-intersecting tiles.
- The group-disjoint frozen-encoder diagnostic produced pooled AUROC 0.7474 and AUPRC 0.5599 over 547 lesion-intersecting and 2,201 sampled background tiles.
- The leave-one-tumor-slide-out full-bag audit produced pooled AUROC 0.7406; all 14 evaluable tumor slides had positive lesion-versus-background probability margins.
- The v4.3 extension produced pooled lesion/background AUROC 0.9014 when both classes occurred, but only 8 of 14 contributing slides contained both lesion and background tiles in that extension.
- Sparse lesion evidence remained a core limitation. Difficult bags included `tumor_002` with one lesion tile and `tumor_012` with two lesion tiles.

## Mechanistic interpretation

The v4.3 sampler finds additional lesion tissue, and the frozen encoder represents lesion-associated morphology. However, lesion evidence is sparse or absent in difficult bags, varies substantially among slides, and is not converted reliably into tumor-versus-normal slide predictions by the tested pooling and tile-ranking methods. The current limitation is therefore not complete encoder failure. It is a combination of sample size, lesion prevalence, slide heterogeneity, aggregation, and calibration.

## Verified evidence archives

The external evidence freeze contains eight checksum-verified archives and a passing `evidence_freeze_receipt.json`. The canonical rerun archive for the LOSO audit has SHA-256 `6d4a098da41ab0afeee46c0ab33b4ed6033155e3e6d317782cbd7dbbe2617fd6`.

The other frozen result hashes are recorded in the external evidence inventory and must remain outside Git.

## Protected boundaries

- Validation embedding arrays loaded: 0.
- CAMELYON16 final-test rerun: false.
- CAMELYON17 execution count: 0.
- Raw annotation XML uploaded to Kaggle: 0.
- No clinical or deployment claim is authorized.
- Any new comparative-performance claim requires a new untouched evaluation cohort.

## Next phase

Start a new modeling phase only after preregistering a genuinely new hypothesis. Preferred directions are:

1. additional development slides or a new untouched cohort;
2. training-only encoder adaptation with slide-disjoint evaluation;
3. spatially structured MIL using coordinate relationships;
4. stain and resolution alignment;
5. one locked development evaluation after model selection is frozen.

Do not continue searching pooling variants on the same 30 training slides.
