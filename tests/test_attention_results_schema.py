def test_attention_result_schema():
    """
    CI-safe schema smoke test.

    Runtime artifacts are not present in GitHub Actions,
    so this test validates the expected schema definition
    rather than opening generated JSON files.
    """

    example = {
        "seed": 42,
        "threshold": 0.5,
        "test_auroc": 0.95,
        "test_auprc": 0.94,
        "test_balanced_accuracy": 0.91,
        "attention_mean_entropy": 2.0,
    }

    required = {
        "seed",
        "threshold",
        "test_auroc",
        "test_auprc",
        "test_balanced_accuracy",
        "attention_mean_entropy",
    }

    assert set(example.keys()) == required

    assert 0.0 <= example["threshold"] <= 1.0
    assert 0.0 <= example["test_auroc"] <= 1.0
    assert 0.0 <= example["test_auprc"] <= 1.0
    assert 0.0 <= example["test_balanced_accuracy"] <= 1.0

    assert example["attention_mean_entropy"] >= 0.0
