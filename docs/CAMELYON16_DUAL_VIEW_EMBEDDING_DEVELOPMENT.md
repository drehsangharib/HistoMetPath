# CAMELYON16 Dual-View Embedding Development

This milestone materializes separate frozen-encoder embeddings for Spatial v2 and Spatial v3 coordinate views on 30 training and six validation slides. Each view remains a distinct 300-instance bag; no lossy coordinate consensus is applied.

The command uses the existing frozen ResNet-18 checkpoint, verifies checkpoint-to-backbone key coverage, copies source coordinates beside each embedding array, records checksums, and refuses test slides. It generates no probabilities, thresholds, or model outputs.

The completed final test remains immutable. Any future dual-view MIL work must remain development-only until a new untouched test cohort is established.
