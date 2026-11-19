import numpy as np
import pytest

from powersim.metrics import accuracy, selection_curve


def test_generator_joint_and_legacy_boundary_are_distinct():
    report = accuracy([[0, 1], [0, 2], [np.nan, 0]], [[0, 0]]*3)
    assert report["success_rate_1mw"] == report["joint_success_rate_1mw"] == pytest.approx(1/3)
    assert report["joint_accuracy_lt_1mw"] == 0
    assert report["generator_accuracy_lt_1mw"] == .5
    assert report["generator_accuracy_count"] == 3
    assert report["generator_accuracy_denominator"] == 6
    assert report["joint_accuracy_denominator"] == 3
    assert report["generator_finite_outputs"] == 5
    assert report["accuracy_samples"] == 2
    assert report["mae_mw"] == .75
    conservative = accuracy([[0, 1], [0, 2], [np.nan, 0]], [[0, 0]]*3, denominator=5)
    assert conservative["generator_accuracy_lt_1mw"] == .3
    assert conservative["success_rate_1mw"] == .2
    assert conservative["per_generator_accuracy_lt_1mw"] == [.4, .2]


def test_empty_accepted_subset_is_not_perfect_accuracy():
    empty = np.empty((0, 3))
    report = accuracy(empty, empty)
    assert report["generator_accuracy_lt_1mw"] is None
    assert report["joint_accuracy_lt_1mw"] is None
    assert report["success_rate_1mw"] is None
    assert report["mae_mw"] is None
    assert report["sources_per_scenario"] == 3
    conservative = accuracy(empty, empty, denominator=4)
    assert conservative["generator_accuracy_lt_1mw"] == conservative["joint_accuracy_lt_1mw"] == 0
    assert conservative["generator_accuracy_denominator"] == 12


def test_nonfinite_outputs_remain_in_accuracy_denominator():
    report = accuracy([[0, np.inf], [np.nan, np.nan]], np.zeros((2, 2)))
    assert report["generator_accuracy_lt_1mw"] == .25
    assert report["joint_accuracy_lt_1mw"] == 0
    assert report["mae_mw"] is None


@pytest.mark.parametrize("prediction,reference,denominator", [
    ([[0, 1]], [[0]], None),
    ([0, 1], [0, 1], None),
    ([[]], [[]], None),
    ([[0]], [[0]], 0),
    ([[0]], [[0]], 1.5),
    ([[0]], [[0]], True),
])
def test_accuracy_rejects_broadcasting_and_invalid_denominators(prediction, reference, denominator):
    with pytest.raises(ValueError):
        accuracy(prediction, reference, denominator=denominator)


def test_selection_curve_uses_supplied_thresholds_and_reports_counts():
    scores = [.1, .2, np.nan]
    prediction, reference = np.array([[0, 0], [2, 0], [0, 0]]), np.zeros((3, 2))
    curve = selection_curve(scores, prediction, reference, [.2, .1, .1])
    assert [row["threshold"] for row in curve] == [None, .1, .2]
    assert curve[0]["accepted_scenarios"] == 0
    assert curve[0]["accepted_accuracy"]["generator_accuracy_lt_1mw"] is None
    assert curve[1]["neural_coverage"] == pytest.approx(1/3)
    assert curve[1]["accepted_accuracy"]["joint_accuracy_lt_1mw"] == 1
    assert curve[2]["accepted_accuracy"]["joint_accuracy_lt_1mw"] == .5
    assert curve[2]["accepted_accuracy"]["generator_accuracy_lt_1mw"] == .75
    assert curve[2]["accepted_accuracy"]["generator_accuracy_denominator"] == 4
    with pytest.raises(ValueError):
        selection_curve(scores, prediction, reference, [np.inf])
