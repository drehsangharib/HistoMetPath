from analysis.lock_camelyon16_development_model import selection_key


def test_model_selection_priority_is_deterministic():
    candidates = [
        {"model_name": "mean", "validation_metrics": {"balanced_accuracy": 0.5, "auroc": 0.8, "auprc": 0.8}},
        {"model_name": "max", "validation_metrics": {"balanced_accuracy": 1.0, "auroc": 0.7, "auprc": 0.7}},
        {"model_name": "attention", "validation_metrics": {"balanced_accuracy": 1.0, "auroc": 0.9, "auprc": 0.85}},
    ]
    selected = max(candidates, key=selection_key)
    assert selected["model_name"] == "attention"
