"""Report regressions, including the hand calculation in the metric contract."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powersim.config import SimulationConfig
from powersim.data import DatasetConfig, generate_dataset
from powersim.output_generation.report_evaluation import comparisons, main
from powersim.output_generation.report_inputs import Inventory
from powersim.output_generation.report_metrics import (Physics, accuracy_summary, change, gate_classification,
                                     gate_mask, physics_summary, ranking_metrics, speedup)


def test_hand_checked_accuracy_aggregations_and_threshold_equality():
    reference = [[10, 20], [10, 20]]
    prediction = [[11, 20], [9.5, 20.5]]
    s = accuracy_summary(prediction, reference, requested=3)
    assert s["element_lt_1mw"] == 3/4
    assert s["max_lt_1mw"] == 1/2
    assert s["mean_lt_1mw"] == 1
    assert s["legacy_joint_le_1mw"] == 1
    assert s["element_lt_0p5mw"] == 1/4
    assert s["max_lt_0p5mw"] == 0
    assert s["mean_lt_0p5mw"] == 0
    assert s["max_lt_2mw"] == 1
    assert s["mae_mw"] == 0.5
    assert s["rmse_mw"] == pytest.approx(np.sqrt(1.5/4))
    assert s["median_absolute_error_mw"] == .5
    assert s["p95_absolute_error_mw"] == pytest.approx(.925)
    assert s["p99_absolute_error_mw"] == pytest.approx(.985)
    assert s["maximum_error_mw"] == 1
    assert s["element_lt_1mw_all_attempts"] == 3/6
    assert s["max_lt_1mw_all_attempts"] == 1/3


def test_nonfinite_accuracy_is_unsuccessful_not_a_zero_error():
    s = accuracy_summary([[2., np.inf], [2., 4.]], [[2., 3.], [2., 3.]], requested=2)
    assert s["complete_error_rows"] == 1
    assert s["element_lt_1mw"] == .5
    assert s["max_lt_1mw"] == 0
    assert s["mae_mw"] == .5


@pytest.fixture
def physics():
    return Physics(
        m_g=np.array([[0., 1.]]), m_d=np.array([[0., 1.]]), constant=np.array([0.]), shunt_mw=1.,
        gen_min=np.array([0., 0.]), gen_max=np.array([10., 10.]),
        branch_min=np.array([-2.]), branch_max=np.array([2.]),
        cp0=np.array([1., 0.]), cp1=np.array([1., 1.]), cp2=np.array([1., 0.]),
    )


def test_raw_balance_bounds_bidirectional_flows_cost_and_missing(physics):
    p = [[6, 5], [2, 10.5], [11.5, -.5], [5, 4.8], [np.nan, 3]]
    x = np.repeat([[5., 5.]], len(p), axis=0)
    result = physics.evaluate(p, x, [48., 45., 50., 0., 50.])
    s = physics_summary(result)
    assert result["balance_residual_mw"] == pytest.approx([0., 1.5, 0., -1.2, np.nan], nan_ok=True)
    assert result["flow_mw"][:4, 0] == pytest.approx([0., 5.5, -5.5, -.2])
    assert result["gen_lower_violation_mw"][2] == pytest.approx([0, .5])
    assert result["gen_upper_violation_mw"][1] == pytest.approx([0, .5])
    assert result["gen_upper_violation_mw"][2] == pytest.approx([1.5, 0])
    assert result["line_violation_mw"][:3, 0] == pytest.approx([0., 3.5, 3.5])
    assert s["balance_pass_rate"] == .4
    assert s["physics_missing"] == 1
    assert s["feasible_rate"] == .2
    assert s["generator_violation_samples"] == 2
    assert s["generator_violation_events"] == 3
    assert s["generator_violation_rate"] == .5
    assert s["generator_aggregate_violation_mw"] == 2.5
    assert s["line_violation_events"] == 2
    assert s["line_distinct_violating_assets"] == 1
    assert s["line_mean_violation_mw"] == s["line_maximum_violation_mw"] == 3.5
    assert s["line_aggregate_violation_mw"] == 7
    assert s["line_limit_plus_1mw_mean_violation_mw"] == 2.5
    assert result["cost_difference_eur_h"][:4] == pytest.approx([0., -27.5, 94.25, 35.8])
    assert np.isnan(result["relative_cost_difference"][3])
    assert s["feasible_cost_difference_eur_h_count"] == 1
    assert s["infeasible_cost_difference_eur_h_count"] == 3
    assert s["infeasible_relative_cost_difference_count"] == 2


def test_balance_and_bound_numerical_tolerances(physics):
    result = physics.evaluate([[6 + 9e-5, 5], [6 + 1.1e-4, 5], [4 - .0009, 7+.0009], [4 - .0011, 7+.0011]],
                              [[5., 5.]]*4, [1.]*4)
    assert result["balance_pass"].tolist() == [True, False, True, True]
    assert result["line_pass"].tolist() == [True, True, True, False]
    assert result["feasible"].tolist() == [True, False, True, False]


def test_classification_positive_is_unreliable_and_unknown_is_excluded():
    accepted = [False, False, True, False, True, True, True]
    reliable = [False, False, False, True, True, True, False]
    result = gate_classification(accepted, reliable, [True]*6+[False])
    assert [result[k] for k in ("tp", "fn", "fp", "tn", "known")] == [2, 1, 1, 2, 6]
    assert result["unreliable_prevalence"] == .5
    assert result["false_acceptance_rate"] == result["false_rejection_rate"] == 1/3
    assert result["precision"] == result["recall"] == 2/3
    assert gate_classification([True], [True], [True])["recall"] is None
    assert gate_mask(np.array([1., 1., np.nan, 1.01]), np.array([True, False, True, True]), 1.).tolist() == [True, False, False, False]


def test_ranking_groups_ties_and_reports_prevalence():
    s = ranking_metrics(np.array([2., 2., 1., 0., np.nan]), np.array([True, False, True, False, True]), np.ones(5, bool))
    assert s["ranking_prevalence"] == .5
    assert s["ranking_n"] == 4
    assert s["auroc"] == .625
    assert s["auprc_ap"] == pytest.approx((.5 + 2/3)/2)
    assert ranking_metrics(np.array([2.]), np.array([True]), np.ones(1, bool))["auroc"] is None


def test_speedup_and_relative_change_require_matching_workload_and_safe_denominator():
    assert speedup(4, .5, "same", "same") == 8
    with pytest.raises(ValueError, match="identical workload"):
        speedup(4, .5, "a", "b")
    with pytest.raises(ValueError, match="positive measured"):
        speedup(4, 0, "same", "same")
    assert change(.6, .5) == pytest.approx((10., 20.))
    assert change(.6, 0) == (60., None)


def test_attack_delta_matches_clean_scenarios_even_when_new_reference_failed(tmp_path):
    common = {"run": "reconstruction", "network": "case30", "profile": "source-informed",
              "method": "M5", "checkpoint": "last", "seed": 27}
    rows = [
        {**common, "experiment_id": "clean", "scenario": "iid_test", "evaluation": "raw", "element_lt_1mw": .5},
        {**common, "experiment_id": "attack", "scenario": "attack1", "evaluation": "raw-fgsm", "element_lt_1mw": 1.},
    ]
    np.savez(tmp_path/"clean.npz", sample_ids=["already-correct", "wrong"],
             predicted_generation_mw=[[0.], [2.]], reference_generation_mw=[[0.], [0.]])
    np.savez(tmp_path/"attack.npz", source_sample_ids=["wrong"])
    bindings = {r["experiment_id"]: {"checkpoint_sha256": "same"} for r in rows}
    [result] = comparisons(rows, bindings, tmp_path)
    assert result["contrast"] == "attack_vs_clean_matched"
    assert result["paired_cases"] == 1
    assert result["element_lt_1mw_baseline"] == 0.
    assert result["element_lt_1mw_difference_pp"] == 100.
    assert result["element_lt_1mw_relative_change_pct"] is None


def test_report_plan_and_reference_only_analysis_do_not_solve(tmp_path, monkeypatch):
    dataset, output = tmp_path/"data", tmp_path/"report"
    generate_dataset(dataset, DatasetConfig(simulation=SimulationConfig(benchmark="case30"),
                                            train_cases=3, validation_cases=1, test_cases=4))

    def unexpected(*args, **kwargs):
        raise AssertionError("Report must not call the optimizer")

    monkeypatch.setattr("powersim.core.solve.solve_reference", unexpected)
    args = ["--artifacts", str(tmp_path), "--reference-dataset", str(dataset),
            "--output", str(output), "--no-figures"]
    assert main([*args, "--plan"]) == 0
    assert not output.exists()
    assert main(args) == 0
    rows = pd.read_csv(output/"metrics.csv")
    assert len(rows) == 3 and rows["requested"].sum() == 12
    assert set(rows["evaluation"]) == {"reference"} and "element_lt_1mw" not in rows
    assert len(pd.read_csv(output/"reference_attempts.csv")) == 12
    with pytest.raises(ValueError, match="will not be overwritten"):
        main(args)


def test_frozen_hash_mismatch_fails_closed(tmp_path):
    file = tmp_path / "artifact.json"
    file.write_text("{}")
    inventory = Inventory(Path(tmp_path), [])
    inventory.expected[file] = "0"*64
    with pytest.raises(ValueError, match="hash mismatch"):
        inventory.checked(file)
