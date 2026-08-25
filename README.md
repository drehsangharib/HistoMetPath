# HistoMetPath

[![CI](https://github.com/drehsangharib/HistoMetPath/actions/workflows/ci.yml/badge.svg)](https://github.com/drehsangharib/HistoMetPath/actions)
[![Development Release](https://img.shields.io/badge/release-development--release--2251265-blue)](https://github.com/drehsangharib/HistoMetPath/releases/tag/development-release-2251265)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reproducible patch-to-slide computational pathology with leakage-safe pseudo-slides, spatial sampling, multiple-instance learning, and frozen evaluation governance.**

HistoMetPath is an open-source research framework for developing slide-level histopathology models from patch-level data while preserving split integrity, spatial diversity, reproducibility, and strict evaluation boundaries. The project combines PatchCamelyon-based development with CAMELYON16 whole-slide workflows and a sealed, currently unexecuted CAMELYON17 external pilot.

> **Current public milestone:** [HistoMetPath Public Development Release](https://github.com/drehsangharib/HistoMetPath/releases/tag/development-release-2251265), published from commit `22512657feb230e50613be7d5b2a9f0624c9461e`.

## Project status

- Public source-only development prerelease published with checksum and audit records.
- 69 automated tests passed at the released commit, with GitHub Actions CI green.
- CAMELYON16 held-out evaluation is complete and immutable; it must not be rerun or used for further tuning.
- CAMELYON17 external pilot is cryptographically sealed and remains unexecuted.
- External execution and real-WSI access remain disabled; execution count is `0 of 1`.
- No definitive external-validation or clinical-performance claim is made.

## Why HistoMetPath

Patch-level datasets are useful for representation learning, but naive patch-to-slide conversion can introduce leakage, spatial bias, optimistic estimates, and repeated evaluation against held-out data. HistoMetPath addresses these risks with explicit data contracts, deterministic sampling, frozen model-selection rules, and auditable execution gates.

## Core capabilities

- Leakage-safe pseudo-slide construction from patch-level data.
- Spatially distributed tile sampling, including frozen spatial-v2 and spatial-v3 development strategies.
- Mean, max, and attention-based multiple-instance learning components.
- Dual-view tile embedding and slide-level aggregation workflows.
- Training-only calibration, selective prediction, uncertainty, and stability analysis.
- CAMELYON16 WSI manifests, lesion-coverage analysis, failure attribution, and reproducible split controls.
- Frozen development decisions and immutable final-test governance.
- External-cohort readiness checks, non-consuming preflights, and a disabled one-time execution engine.
- Automated tests, CI, manifests, checksums, documentation, and source-only release auditing.

## Pipeline overview

```text
Patch-level data / WSI manifests
        |
Leakage-safe split and pseudo-slide construction
        |
Spatial tile sampling and tissue-quality controls
        |
Frozen tile embeddings and MIL aggregation
        |
Slide-level calibration, uncertainty, and stability audits
        |
Immutable held-out evaluation and sealed external execution contract
```

## Evaluation governance

### CAMELYON16

The completed held-out CAMELYON16 test is immutable. The recorded result is a small development benchmark and must not be rerun, selectively reported, or used for additional tuning. The release preserves this boundary through frozen configurations and decision records.

### CAMELYON17

The CAMELYON17 external pilot is sealed at the released commit and remains unexecuted. The pilot permits a maximum of one deliberate execution, and routine development or documentation work must not consume that evaluation. Any future result must be described as a constrained pilot rather than definitive external or clinical validation.

## Reproducibility and testing

Install dependencies and run the test suite:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pytest tests -v
```

The repository uses deterministic manifests, fixed seeds where applicable, configuration-driven workflows, GitHub Actions CI, and checksum-backed release artifacts. Raw WSI data, embeddings, checkpoints, and runtime outputs are intentionally excluded from version control.

## Repository structure

```text
.github/workflows/   GitHub Actions CI
analysis/            modeling, calibration, WSI, readiness, and audit commands
configs/             frozen experiment and evaluation contracts
core/                reusable pipeline and WSI components
docs/                development decisions and execution-governance documentation
models/              patch-level and MIL model definitions
scripts/             acquisition and operational utilities
tests/               automated unit, smoke, schema, and safety tests
training/            training workflows
```

## Public development release

The reviewed source-only prerelease includes:

- publication-ready source ZIP;
- SHA-256 checksum sidecar;
- final publication audit JSON.

Release: https://github.com/drehsangharib/HistoMetPath/releases/tag/development-release-2251265

## Scientific scope and limitations

HistoMetPath is a research and software-engineering framework. It is not a medical device, is not clinically validated, and is not intended for patient-care decisions. Pseudo-slide results must not be interpreted as native whole-slide or patient-level diagnostic performance. New comparative claims require a new untouched evaluation cohort.

## Citation

Please use the repository's `CITATION.cff` file and cite the exact release tag used in your work.

## License

Released under the [MIT License](LICENSE).
