# CAMELYON16 Development-Only Model Lock

This stage loads only the 30 training and six validation slide bags. It refuses to load fresh-test embeddings.

Mean pooling, max pooling, and a predeclared five-seed Attention MIL ensemble are compared using validation balanced accuracy, then validation AUROC, validation AUPRC, and model name as deterministic tie-breakers. The selected model, preprocessing parameters, probability threshold, model state, and input checksums are serialized into a development artifact.

The six fresh test slides remain untouched. A separate final-test command must verify the development artifact and lock checksums before evaluating them exactly once.
