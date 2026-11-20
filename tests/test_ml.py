import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powersim.data import DatasetConfig, generate_dataset, load_dataset
from powersim.dc import DCRelation, reference_jacobian
from powersim.experiments import accuracy, evaluate_model
from powersim.hybrid import HybridPredictor
from powersim.ml import (
    OPFNetwork, TrainConfig, Decoder, artifact_hash, calibrate_threshold, jacobian_losses,
    load_decoder, load_model, physical_jacobian, recalibrate_decoder, tensor, train_decoder, train_model,
)
from powersim.solver import SolveResult


def small_model():
    return OPFNetwork([0, 0], [2, 4], [0, 0], [3, 5], hidden=(3, 3, 3))


def test_normalized_chain_rule_and_second_derivatives():
    torch.manual_seed(2)
    model = small_model()
    x = tensor([[2, 3], [4, 5]]).requires_grad_(True)
    j = physical_jacobian(model, x)
    normalized_input = ((x-model.x_mean)/model.x_scale).detach().requires_grad_(True)
    normalized_output = model.normalized_forward(normalized_input)
    normalized_j = torch.stack([torch.autograd.grad(normalized_output[:, g].sum(), normalized_input,
                              create_graph=True, retain_graph=True)[0] for g in range(2)], dim=1)
    torch.testing.assert_close(j, normalized_j*model.y_scale[None, :, None]/model.x_scale[None, None, :])
    (j.sum(dim=1)-1).square().mean().backward()
    assert model.layers[0].weight.grad is not None
    assert model.layers[0].weight.grad.abs().sum() > 0


def test_per_unit_relu_jacobians_have_trainable_mixed_parameter_derivatives():
    model = OPFNetwork([0, 0], [100, 100], [0, 0], [100, 100],
                       hidden=(3, 3, 3), activation="relu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.2)  # Positive preactivations, away from ReLU kinks.
    x = tensor([[50, 60]]).requires_grad_(True)
    j = physical_jacobian(model, x)
    z = (x/100).detach().requires_grad_(True)
    y_pu = model.normalized_forward(z)
    j_pu = torch.stack([torch.autograd.grad(y_pu[:, g].sum(), z, create_graph=True,
                       retain_graph=True)[0] for g in range(2)], dim=1)
    torch.testing.assert_close(j, j_pu)  # Common input/output unit base cancels.
    hessian_sum = torch.autograd.grad(j.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(hessian_sum, torch.zeros_like(x))
    (j.sum(dim=1)-1).square().mean().backward()
    assert model.layers[0].weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0


def test_jacobian_identities_and_only_local_active_masks():
    j = tensor([[[.6, .6], [.4, .4]]]).requires_grad_(True)
    mg = tensor([[1, 0], [0, 1]])
    md = tensor([[.6, .6], [.4, .4]])
    losses = jacobian_losses(j, j.detach(), tensor([1]).bool(), tensor([[0, 0]]).bool(),
                            tensor([[1, 0]]).bool(), mg, md)
    assert all(float(v.detach()) == pytest.approx(0) for v in losses.values())
    inactive = jacobian_losses(j, j.detach()+1, tensor([0]).bool(), tensor([[0, 0]]).bool(),
                              tensor([[0, 0]]).bool(), mg, md+3)
    assert inactive["L1"] == inactive["L3"] == inactive["L4"] == 0
    active = jacobian_losses(j, j.detach(), tensor([1]).bool(), tensor([[1, 0]]).bool(),
                            tensor([[1, 0]]).bool(), mg, md+1)
    assert active["L3"] > 0 and active["L4"] > 0
    (active["L3"]+active["L4"]).backward()
    assert j.grad.abs().sum() > 0


def test_singular_sensitivity_unavailable():
    relation = DCRelation(np.ones((1, 2)), np.ones((1, 2)), np.zeros(1), ("line:0",), 0)
    j, condition = reference_jacobian([.1, .1], relation, [False, False], [False, False], [True])
    assert j is None  # active branch equality duplicates balance


def test_decoder_routing_failed_fallback_and_production_log(tmp_path):
    model = small_model()
    decoder = Decoder(3, 2, 4)
    metadata = {"threshold": 1e9, "error_units": "normalized"}
    predictor = HybridPredictor(model, decoder, metadata, ["ext_grid:0", "gen:0"])
    def not_called():
        raise AssertionError("Accepted samples must not run OPF")
    r = predictor.predict([0, 0], not_called)
    assert r.status == "learned" and r.accepted
    metadata["threshold"] = 0
    calls = []
    def fallback():
        calls.append(True)
        return SolveResult("ok", True, "dc", "opf", .1, generation_mw={"ext_grid:0": 1, "gen:0": 2},
                           metrics={"feasible": True})
    r = predictor.predict([20, 30], fallback, production_log=tmp_path/"production.jsonl")
    assert r.status == "fallback" and r.generation_mw == (1, 2) and calls == [True]
    assert r.raw_prediction_mw != r.generation_mw
    assert json.loads((tmp_path/"production.jsonl").read_text())["net_load_mw"] == [20, 30]
    r = predictor.predict([20, 30], lambda: SolveResult("failed", False, "dc", "opf", .1))
    assert r.status == "failed" and r.generation_mw is None
    assert predictor.predict([float("nan"), 0], not_called).status == "failed"


def test_calibration_and_success_definition():
    assert calibrate_threshold([1, 2, 3, 4], .5) == 3
    with pytest.raises(ValueError):
        calibrate_threshold([float("nan")])
    report = accuracy([[0, 1], [0, 2], [np.nan, 0]], [[0, 0]]*3)
    assert report["success_rate_1mw"] == pytest.approx(1/3)
    assert report["maximum_error_mw"] == 2


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("ml") / "data"
    generate_dataset(path, DatasetConfig(train_cases=8, validation_cases=3, test_cases=3))
    return path


def test_short_cpu_training_decoder_and_evaluation(dataset, tmp_path):
    cfg = TrainConfig(hidden=(8, 8, 8), epochs=2, batch_size=4,
                      loss_weights={f"L{i}": 1.0 for i in range(6)})
    report = train_model(dataset, tmp_path/"physics", cfg)
    assert np.isfinite(report["best_validation_mse_mw2"])
    assert all(np.isfinite(list(h.values())).all() for h in report["history"])
    model_path = tmp_path/"physics"/"model.pt"
    before = artifact_hash(model_path)
    meta = train_decoder(dataset, model_path, tmp_path/"decoder", epochs=2, hidden=8)
    assert artifact_hash(model_path) == before
    assert meta["opf_frozen"] and meta["calibration_split"] == "validation"
    model, _ = load_model(model_path)
    decoder, _ = load_decoder(tmp_path/"decoder"/"decoder.pt", model_path, model)
    calibration = recalibrate_decoder(dataset, model_path, tmp_path/"decoder"/"decoder.pt",
                                      tmp_path/"calibrated", quantile=.5, units="mw2")
    assert calibration["error_units"] == "mw2"
    assert calibration["calibration_split"] == "validation"
    assert model.features(tensor([[0]*30])).shape == (1, 8)
    report = evaluate_model(dataset, model_path, tmp_path/"evaluation",
                            decoder_path=tmp_path/"decoder"/"decoder.pt")
    assert 0 <= report["decoder_acceptance_rate"] <= 1
    assert report["timing"]["single_scenario_mean_seconds"] > 0
    assert report["raw_physics"]["voltage_violations"] is None
    assert (tmp_path/"evaluation"/"predictions.jsonl").exists()
    parallel = evaluate_model(dataset, model_path, tmp_path/"parallel", workers=2,
                              decoder_path=tmp_path/"decoder"/"decoder.pt")
    for metric in ("raw_accuracy", "hybrid_accuracy", "raw_physics", "hybrid_physics",
                   "decoder_acceptance_rate", "fallback_rate", "accepted_accuracy",
                   "fallback_accuracy", "accepted_physics", "coverage", "selection_curve",
                   "raw_all_requested_accuracy", "hybrid_all_requested_accuracy"):
        assert report[metric] == parallel[metric]
    assert parallel["timing"]["single_scenario_mean_seconds"] is None
    assert parallel["timing"]["evaluation_cases_per_second_including_physical_audits"] > 0
    serial_ids = [json.loads(row)["scenario_id"] for row in (tmp_path/"evaluation"/"predictions.jsonl").read_text().splitlines()]
    parallel_ids = [json.loads(row)["scenario_id"] for row in (tmp_path/"parallel"/"predictions.jsonl").read_text().splitlines()]
    assert serial_ids == parallel_ids
    with pytest.raises(ValueError, match="held-out"):
        evaluate_model(dataset, model_path, tmp_path/"invalid", split="validation")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py", default_timeout=20).run()
    app.sidebar.number_input[0].set_value(.9).run()
    app.sidebar.text_input[0].set_value(str(model_path)).run()
    app.sidebar.text_input[1].set_value(str(tmp_path/"decoder"/"decoder.pt")).run()
    assert not app.exception and not app.warning
    assert "learned" in app.sidebar.selectbox[2].options
    app.sidebar.selectbox[2].select("learned").run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["result"]
    assert result["status"] in ("learned", "fallback")
    assert result["raw_evaluation"]["operation"] == "pf"
    assert result["bus_voltage_pu"] is None
    app.sidebar.selectbox(key="physics").select("ac").run()
    assert "learned" not in app.sidebar.selectbox[2].options


def test_baseline_deterministic_and_test_labels_unused(dataset, tmp_path):
    cfg = TrainConfig(hidden=(8, 8, 8), epochs=1)
    train_model(dataset, tmp_path/"baseline1", cfg)
    # Alter held-out labels only, keeping every training/validation value fixed.
    import shutil
    changed = tmp_path/"changed"
    shutil.copytree(dataset, changed)
    manifest, arrays = load_dataset(changed)
    for split in manifest["splits"]:
        if split not in ("train", "validation"):
            arrays[f"{split}_y"] += 10000
            arrays[f"{split}_x"] += 10000
    np.savez_compressed(changed/"dataset.npz", **arrays)
    train_model(changed, tmp_path/"baseline2", cfg)
    _, c1 = load_model(tmp_path/"baseline1"/"model.pt")
    _, c2 = load_model(tmp_path/"baseline2"/"model.pt")
    for key, value in c1["state_dict"].items():
        torch.testing.assert_close(value, c2["state_dict"][key], rtol=0, atol=0)
    model_path = tmp_path/"baseline1"/"model.pt"
    d1 = train_decoder(dataset, model_path, tmp_path/"decoder1", epochs=1, hidden=4)
    d2 = train_decoder(changed, model_path, tmp_path/"decoder2", epochs=1, hidden=4)
    assert d1["threshold"] == d2["threshold"]
    model, _ = load_model(model_path)
    decoder1, _ = load_decoder(tmp_path/"decoder1"/"decoder.pt", model_path, model)
    decoder2, _ = load_decoder(tmp_path/"decoder2"/"decoder.pt", model_path, model)
    for key, value in decoder1.state_dict().items():
        torch.testing.assert_close(value, decoder2.state_dict()[key], rtol=0, atol=0)
