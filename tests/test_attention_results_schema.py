import json

def test_attention_result_schema():
    with open(
        "outputs/mil_controlled_attention/attention_results.json",
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    assert len(results) == 5

    for result in results:
        assert "seed" in result
        assert "test_auroc" in result
        assert "test_auprc" in result
        assert "attention_mean_entropy" in result
