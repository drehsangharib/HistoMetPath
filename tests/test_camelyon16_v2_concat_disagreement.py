from analysis.audit_camelyon16_v2_concat_disagreement import summarize_probabilities


def test_probability_summary():
    result = summarize_probabilities([0.1, 0.3, 0.5])
    assert result["mean"] == 0.3
    assert result["minimum"] == 0.1
    assert result["maximum"] == 0.5
    assert result["range"] == 0.4
