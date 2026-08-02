# Frozen External One-Time Runner

This milestone adds the reviewed runner contract and a strict non-consuming preflight mode. The execution implementation is intentionally disabled. This ensures the only authorized CAMELYON17 pilot execution cannot be consumed accidentally while the external processing adapter is still undergoing source review.

The scaffold validates the sealed lock, checksum record, readiness result, prior non-consuming preflight, frozen artifact, frozen threshold, feature dimensions, cohort counts, Git commit, clean working tree, and available disk space. `--execute` always refuses and leaves the execution count unchanged, even when the explicit token is supplied.

A later activation milestone must implement and test the unchanged Spatial v2/v3 sampling adapter, frozen encoder embedding path, frozen model inference, pre-processing outputs, interruption semantics, execution-count transition, and immediate result sealing. That activation must be committed, CI-green, resealed, and preflighted before the single execution is attempted.
