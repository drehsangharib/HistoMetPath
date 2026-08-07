# Frozen External Execution Engine — Synthetic Dry Run

This milestone validates the final one-time execution engine using synthetic embeddings and temporary lock fixtures only. Real WSI access and external execution remain disabled. The dry run verifies 300x512 per-view embedding contracts, 512-feature primary pooling, 1024-feature dual-view concatenation, frozen scaler/model inference, the frozen primary threshold, atomic fixture-state writes, consumption-before-processing semantics, sealed partial-fixture receipts, and permanent refusal of a second fixture execution.

The module never opens an external WSI, never calls a sampler, never transforms a real tile, never writes an external embedding or prediction, never mutates the real lock, and never consumes the authorized external execution.
