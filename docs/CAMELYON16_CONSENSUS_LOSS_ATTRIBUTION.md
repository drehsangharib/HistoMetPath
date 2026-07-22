# CAMELYON16 Consensus Loss Attribution

This development-only audit measures the full v2/v3 coordinate union as an upper bound and explains lesion coverage lost when the union is reduced to the frozen 300-tile consensus.

For each annotated development tumor slide, the audit classifies lesion coordinates as shared, v2-only, or v3-only; records whether each coordinate was retained by consensus; quantifies discarded lesion tiles; measures union versus consensus polygon coverage; and identifies parent-supported bags lost under the cap.

The audit does not create a new sampler, load test slides, generate embeddings, or produce model outputs. The completed final test remains immutable.
