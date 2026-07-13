# CAMELYON16 22-Slide Batch Pilot

This configuration-driven pipeline discovers 22 local CAMELYON16 training WSIs, assigns a deterministic stratified slide-level split, chooses a pyramid level by physical resolution, samples tissue-rich tiles, and extracts trained CNN embeddings without saving additional PNG tiles.

## Pilot split

- Train: 7 normal + 7 tumor
- Validation: 2 normal + 2 tumor
- Test: 2 normal + 2 tumor

## Processing design

- Target resolution: approximately 0.5 microns/pixel
- Tile size: 256 x 256 pixels
- Maximum accepted tiles: 300 per slide
- Tissue threshold and all processing settings are recorded in YAML
- Coordinates remain in level-zero reference space
- Runtime outputs and embeddings remain ignored by Git

## Scientific scope

This is a small real-WSI pilot. It is suitable for validating end-to-end mechanics and an initial held-out experiment, but not for definitive clinical-performance claims.
