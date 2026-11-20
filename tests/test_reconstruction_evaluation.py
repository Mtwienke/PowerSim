import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powersim.attacks import fgsm_inputs, generate_fgsm_dataset
from powersim.data import DatasetConfig, generate_dataset, load_dataset
from powersim.experiments import _physical_summary, evaluate_model
from powersim.metrics import accuracy
from powersim.ml import TrainConfig, artifact_hash, train_decoder, train_model
from powersim.solver import BranchResult, SolveResult


def test_fgsm_direction_budget_and_no_input_clipping():
    class LinearModel(torch.nn.Module):
        y_scale = torch.ones(1)

        def forward(self, x):
            return x[:, :1]+2*x[:, 1:]

    model = LinearModel()
    x = np.array([[1., 1.], [0., -.5]])
    target = np.zeros((2, 1))
    attacked, delta = fgsm_inputs(model, x, target, TrainConfig(), epsilon_mw=1.)
    np.testing.assert_array_equal(delta, [[1, 1], [-1, -1]])
    np.testing.assert_array_equal(attacked, x+delta)
    assert attacked[1, 1] < 0  # No hidden clipping of net injections.
    with pytest.raises(ValueError, match="epsilon"):
        fgsm_inputs(model, x, target, TrainConfig(), epsilon_mw=-1)


def test_source_branch_metric_counts_events_above_limit_plus_one_without_weakening_strict():
    def result(flow):
        return SolveResult("ok", True, "dc", "pf", .01,
                           branches=(BranchResult("line:0", flow, -flow, 10., flow*10, 100, False, flow > 10.001),),
                           metrics={"feasible": False, "branch_violations": int(flow > 10.001)},
                           raw_balance_residual_mw=2.)
    report = _physical_summary([result(10.5), result(11.5)], [1, 1])
    assert report["branch_violations"] == 2
    assert report["source_like_branch_events_beyond_limit_plus_1mw"] == 1
    assert report["source_like_mean_excess_beyond_limit_plus_1mw"] == .5
    assert report["feasible_rate"] == 0


def test_scenario_mean_metric_is_distinct_and_failed_outputs_stay_failures():
    report = accuracy([[0, 1.5], [0, np.nan]], [[0, 0], [0, 0]], denominator=3)
    assert report["scenario_mae_accuracy_lt_1mw"] == pytest.approx(1/3)
    assert report["joint_accuracy_lt_1mw"] == 0
    assert report["generator_accuracy_lt_1mw"] == pytest.approx(2/6)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reconstruction-eval")
    dataset = directory/"data"
    generate_dataset(dataset, DatasetConfig(train_cases=4, validation_cases=2, test_cases=3,
                                            sampling_policy="valid"))
    model = directory/"model"
    train_model(dataset, model, TrainConfig(hidden=(8, 8, 8), epochs=1))
    decoder = directory/"decoder"
    train_decoder(dataset, model/"model.pt", decoder, epochs=1, hidden=(8, 8), units="rmse_mw", threshold=6)
    return dataset, model/"model.pt", decoder/"decoder.pt"


def test_compressed_evaluations_and_predeclared_norm_diagnostics(source, tmp_path):
    dataset, model, decoder = source
    report = evaluate_model(dataset, model, tmp_path/"eval", decoder_path=decoder,
                            prediction_format="jsonl.gz", decoder_norm_sensitivity=True)
    assert report["prediction_file"] == "predictions.jsonl.gz"
    assert not (tmp_path/"eval/predictions.jsonl").exists()
    with gzip.open(tmp_path/"eval/predictions.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == report["labeled_cases"] == 3
    diagnostics = report["decoder_norm_sensitivity"]["scores"]
    assert set(diagnostics) == {"mae_mw", "rmse_mw", "l1_mw", "linf_mw"}
    assert diagnostics["rmse_mw"]["thresholds"]["fixed_6mw"] == 6


def test_attack_references_are_resolved_at_new_inputs_and_fixed_targets_separate(source, tmp_path, monkeypatch):
    dataset, model, _ = source
    before = artifact_hash(model)
    import powersim.attacks as module
    called = []
    original = module._solve_scenario

    def record(record):
        called.append(record["net_load_mw"])
        return original(record)

    monkeypatch.setattr(module, "_solve_scenario", record)
    report = generate_fgsm_dataset(dataset, model, tmp_path/"attack", epsilon_mw=1.)
    assert report["original_scenarios"] == len(called) == 3
    manifest, attacked = load_dataset(tmp_path/"attack")
    _, old = load_dataset(dataset)
    np.testing.assert_array_equal(attacked["attack_all_original_y"], old["iid_test_y"])
    assert manifest["evaluation_only"] is True
    assert report["fixed_original_target_accuracy_all_attacks"]["evaluated_scenarios"] == 3
    assert report["maximum_linf_mw"] <= 1+1e-9
    np.testing.assert_allclose(np.asarray(called), attacked["attack_all_x"])
    assert artifact_hash(model) == before
    if report["new_reference_valid"]:
        evaluation = evaluate_model(tmp_path/"attack", model, tmp_path/"eval-attack", split="attack1")
        assert evaluation["labeled_cases"] == report["new_reference_valid"]
        assert evaluation["raw_accuracy"]["evaluated_scenarios"] == report["new_reference_valid"]
    with pytest.raises(ValueError, match="held-out"):
        generate_fgsm_dataset(dataset, model, tmp_path/"forbidden", split="train")
    assert not (tmp_path/"forbidden").exists()
