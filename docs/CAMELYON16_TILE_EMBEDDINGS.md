# CAMELYON16 Real Tile-Embedding Pilot

This milestone extracts one feature vector per audited tissue tile from real CAMELYON16 whole-slide images.

## Inputs

- `normal_100.tif`
- `tumor_100.tif`
- Tissue-tile manifests that passed the tile audit
- A trained HistoMetPath patch-model checkpoint

## Outputs

For each slide:

- `<slide>_embeddings.npy`
- `<slide>_coordinates.npy`
- `<slide>_tile_names.json`

A global `outputs/camelyon16/embedding_manifest.json` records the checkpoint checksum, extraction configuration, dimensions, counts, and output checksums.

## Scientific Scope

This is a two-slide real-WSI pipeline validation. It demonstrates WSI reading, tissue filtering, tile extraction, and real tile-embedding generation. It is not a model-performance benchmark and does not support generalizable classification claims.
