# Spatial-v3.1 Operational Controls

## Lineage

- Canonical algorithm: Spatial-v3.
- Qualified implementation source SHA-256: $ExpectedSourceHash.
- Designation: Spatial-v3.1 operational-control instrumentation.
- This is not a historical byte-identical replay.

## Added controls

- Exact attempted and successful region-read accounting.
- Final int64 coordinate and uniqueness validation.
- Atomic per-slide output promotion.
- Per-slide receipts and array SHA-256 values.
- Restart-safe reuse and corruption rejection.
- Import isolation from torch-based training modules.

## Qualification evidence

- Expanded synthetic suite: 11 tests passed.
- Real-slide qualification: normal_027 training WSI only.
- Two independent runs produced 300 unique coordinates each.
- Coordinates and tissue fractions were byte-reproducible.
- Restart reuse opened zero WSIs and performed zero region reads.
- Corrupted copied output was rejected.
- Validation, protected test, CAMELYON17, annotations, embeddings, training, and evaluation remained zero.

## Full-cohort boundary

A full 30-slide Spatial-v3.1 rerun was not performed because nine frozen historical training WSIs were unavailable in the controlled Kaggle environment. Existing Spatial-v3 full-cohort coordinate and embedding evidence is not relabeled as Spatial-v3.1 output.

## Accelerator policy

Coordinate generation, static tests, integrity audits, and repository tests require no GPU. Frozen ResNet-18 embedding extraction uses GPU.
