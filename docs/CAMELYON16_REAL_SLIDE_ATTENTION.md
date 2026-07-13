# CAMELYON16 Real Slide-Bag Attention Pilot

This milestone validates variable-size real slide bags, coordinate alignment, top-attended tile retrieval, and WSI overlay generation for `normal_100` and `tumor_100`.

The Attention MIL model used for this mechanics pilot is trained on controlled synthetic PCAM bags and then applied to the two real CAMELYON16 slide bags. Consequently, exploratory probabilities and attention weights are not validated real-WSI predictions, lesion annotations, or clinical outputs.

Reliable real-WSI performance evaluation requires a larger CAMELYON16 cohort and slide-level train, validation, and test partitions.
