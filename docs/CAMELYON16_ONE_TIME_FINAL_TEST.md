# CAMELYON16 One-Time Fresh-Test Gate

This command evaluates the checksum-locked development model on exactly six predeclared fresh test slides. It verifies the development counts, untouched boundary, selected model, threshold, artifact checksum, processing-manifest checksum, holdout identities, and test identities before loading test embeddings.

The evaluator refuses to overwrite an existing result and creates `FINAL_TEST_EXECUTED.lock` after a successful run.

The test set contains only six slides; one prediction changes accuracy by 16.7 percentage points. Results remain preliminary and are not clinical-performance estimates.
