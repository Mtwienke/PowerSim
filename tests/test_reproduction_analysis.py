import json

import pytest

pytest.importorskip("torch")

from powersim.research.reproduction_analysis import analyze_suite, pooled_reports


def test_pooled_accuracy_uses_counts_and_rmse_uses_squared_errors():
    first = {
        "labeled_cases": 1,
        "raw_accuracy": {"generator_accuracy_count": 1, "generator_accuracy_denominator": 2,
                         "joint_accuracy_count": 0, "mae_mw": 1., "rmse_mw": 2.},
        "raw_physics": {"feasible_predictions": 0, "balance_mean_absolute_mw": 3.,
                        "source_like_branch_events_beyond_limit_plus_1mw": 2},
    }
    second = {
        "labeled_cases": 3,
        "raw_accuracy": {"generator_accuracy_count": 6, "generator_accuracy_denominator": 6,
                         "joint_accuracy_count": 3, "mae_mw": 2., "rmse_mw": 4.},
        "raw_physics": {"feasible_predictions": 3, "balance_mean_absolute_mw": 1.,
                        "source_like_branch_events_beyond_limit_plus_1mw": 1},
    }
    result = pooled_reports([first, second])
    assert result["generator_accuracy_lt_1mw"] == 7/8
    assert result["joint_accuracy_lt_1mw"] == result["feasible_rate"] == 3/4
    assert result["mae_mw"] == 7/4
    assert result["rmse_mw"] == pytest.approx(13**.5)
    assert result["source_like_branch_events_beyond_limit_plus_1mw"] == 3


def test_analysis_does_not_read_partial_suite_results_or_overwrite_outputs(tmp_path):
    root = tmp_path/"suite"
    root.mkdir()
    (root/"status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="completed"):
        analyze_suite(root)
    assert not (root/"analysis").exists()
    output = tmp_path/"existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        analyze_suite(root, output)
