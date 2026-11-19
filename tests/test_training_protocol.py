from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powersim.cli import main
from powersim.data import DatasetConfig, generate_dataset, load_dataset
from powersim.experiments import evaluate_model
from powersim.ml import (
    Decoder, METHOD_WEIGHTS, TrainConfig, artifact_hash, load_decoder, load_model,
    reconstruction_errors, state_dict_hash, tensor, train_decoder, train_model, training_data_hash,
)
from powersim.metrics import accuracy
from powersim.pilot import PilotConfig, pilot_plan, run_pilot


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("protocol") / "data"
    generate_dataset(directory, DatasetConfig(train_cases=8, validation_cases=3, test_cases=3))
    return directory


@pytest.fixture(scope="module")
def parent(dataset, tmp_path_factory):
    directory = tmp_path_factory.mktemp("protocol-model") / "M1"
    train_model(dataset, directory, TrainConfig(hidden=(8, 8, 8), epochs=2, checkpoint_selection="last"))
    return directory / "model.pt"


def test_verified_method_taxonomy_and_pilot_defaults():
    assert METHOD_WEIGHTS["M1"] == {"L0": 1.0}
    assert METHOD_WEIGHTS["M2"] == {"L0": 1.0, "L1": 1.0}
    assert set(METHOD_WEIGHTS["M4"]) == {f"L{i}" for i in range(5)}
    assert METHOD_WEIGHTS["M5"] == METHOD_WEIGHTS["physics-guided"]
    assert "M3" not in METHOD_WEIGHTS
    config = PilotConfig()
    assert (config.pretrain_epochs, config.fine_tune_epochs, config.learning_rates) == (1000, 200, (.01,))
    assert config.checkpoint_selection == "last"


def test_warm_start_inherits_exact_weights_and_keeps_best_and_last(dataset, parent, tmp_path):
    before = artifact_hash(parent)
    model, source = load_model(parent)
    report = train_model(dataset, tmp_path/"M2", TrainConfig(
        hidden=(8, 8, 8), epochs=2, loss_weights=METHOD_WEIGHTS["M2"]), initialize_from=parent)
    assert report["initialization"]["checkpoint_sha256"] == before
    assert report["initialization"]["state_dict_sha256"] == state_dict_hash(source["state_dict"])
    assert report["initialization"]["training_data_hash_verified"] is True
    assert report["initialization"]["parent_epoch"] == 2
    assert "Fresh Adam" in report["initialization"]["optimizer_state"]
    assert artifact_hash(parent) == before
    _, arrays = load_dataset(dataset)
    with torch.no_grad():
        initial = accuracy(model(tensor(arrays["validation_x"])).numpy(), arrays["validation_y"])
    assert report["initial_validation_accuracy"] == initial
    _, selected = load_model(tmp_path/"M2"/"model.pt")
    _, best = load_model(tmp_path/"M2"/"best.pt")
    _, last = load_model(tmp_path/"M2"/"last.pt")
    assert selected["state_dict_sha256"] == best["state_dict_sha256"]
    assert selected["epoch"] == report["best_epoch"]
    assert last["epoch"] == 2 and last["checkpoint_selection"] == "last"
    assert report["validation_accuracy"]["joint_accuracy_denominator"] > 0
    assert report["training_accuracy"]["generator_accuracy_denominator"] > 0
    for file, recorded in report["checkpoints"].items():
        assert artifact_hash(tmp_path/"M2"/file) == recorded["sha256"]


@pytest.mark.parametrize("field,message", [
    ("model_id", "dataset/model/ordering"),
    ("bus_ids", "dataset/model/ordering"),
    ("hidden", "architecture"),
    ("normalization", "normalization"),
    ("training_input_ids", "inputs"),
    ("training_data_sha256", "data"),
])
def test_incompatible_initializers_fail_before_creating_outputs(dataset, parent, tmp_path, field, message):
    checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    if field == "hidden":
        checkpoint["configuration"]["hidden"] = [9, 9, 9]
    elif field == "normalization":
        checkpoint["state_dict"]["x_scale"] += 1
    elif field == "bus_ids":
        checkpoint[field] = checkpoint[field][::-1]
    elif field == "training_input_ids":
        checkpoint[field][0] = "changed"
    else:
        checkpoint[field] = "changed"
    source = tmp_path / "incompatible.pt"
    torch.save(checkpoint, source)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        train_model(dataset, output, TrainConfig(hidden=(8, 8, 8), epochs=1), initialize_from=source)
    assert not output.exists()


def test_held_out_changes_do_not_affect_warm_start_data_identity(dataset, parent, tmp_path):
    changed = tmp_path/"changed"
    shutil.copytree(dataset, changed)
    _, arrays = load_dataset(changed)
    original_hash = training_data_hash(arrays)
    arrays["ood_test_2_y"] += 10000
    arrays["ood_test_2_x"] += 10000
    assert training_data_hash(arrays) == original_hash
    np.savez_compressed(changed/"dataset.npz", **arrays)
    config = TrainConfig(hidden=(8, 8, 8), epochs=1)
    first = train_model(dataset, tmp_path/"first", config, initialize_from=parent)
    second = train_model(changed, tmp_path/"second", config, initialize_from=parent)
    assert first["initialization"] == second["initialization"]
    _, a = load_model(tmp_path/"first"/"model.pt")
    _, b = load_model(tmp_path/"second"/"model.pt")
    assert a["state_dict_sha256"] == b["state_dict_sha256"]
    arrays["validation_y"] += 1
    np.savez_compressed(changed/"dataset.npz", **arrays)
    with pytest.raises(ValueError, match="data mismatch"):
        train_model(changed, tmp_path/"invalid", config, initialize_from=parent)


def test_loss_ablations_share_shuffle_stream(dataset, tmp_path, monkeypatch):
    original = torch.randperm
    permutations = []

    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        permutations.append(value.clone())
        return value

    monkeypatch.setattr(torch, "randperm", record)
    train_model(dataset, tmp_path/"baseline", TrainConfig(hidden=(8, 8, 8), epochs=2))
    baseline = permutations[:]
    permutations.clear()
    train_model(dataset, tmp_path/"guided", TrainConfig(
        hidden=(8, 8, 8), epochs=2, loss_weights=METHOD_WEIGHTS["M5"]))
    assert len(permutations) == len(baseline) == 2
    for a, b in zip(permutations, baseline):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_decoder_physical_scores_have_correct_units():
    model = SimpleNamespace(x_mean=tensor([0, 0]), x_scale=tensor([2, 4]), features=lambda x: x)
    decoder = lambda x: torch.zeros_like(x)
    x = tensor([[2, 8]])
    assert reconstruction_errors(model, decoder, x, "normalized").item() == 2.5
    assert reconstruction_errors(model, decoder, x, "mw2").item() == 34
    assert reconstruction_errors(model, decoder, x, "mae_mw").item() == 5
    assert reconstruction_errors(model, decoder, x, "rmse_mw").item() == pytest.approx(np.sqrt(34))
    with pytest.raises(ValueError):
        reconstruction_errors(model, decoder, x, "unknown")


def test_two_hidden_layer_decoder_and_explicit_threshold(dataset, parent, tmp_path):
    before = artifact_hash(parent)
    metadata = train_decoder(dataset, parent, tmp_path/"decoder", hidden=(8, 4), epochs=2,
                             units="rmse_mw", threshold=6.)
    assert metadata["hidden_layers"] == [8, 4]
    assert metadata["score_units"] == "MW"
    assert metadata["threshold"] == 6 and metadata["quantile"] is None
    assert metadata["calibration_method"] == "explicit_fixed_threshold"
    assert len(metadata["validation_thresholds"]) == 9
    model, _ = load_model(parent)
    decoder, _ = load_decoder(tmp_path/"decoder"/"decoder.pt", parent, model)
    assert sum(isinstance(layer, torch.nn.Linear) for layer in decoder.layers) == 3
    assert artifact_hash(parent) == before
    with pytest.raises(ValueError):
        train_decoder(dataset, parent, tmp_path/"invalid", epochs=1, threshold=float("nan"))
    assert not (tmp_path/"invalid").exists()


def test_legacy_decoder_checkpoint_is_still_loadable(dataset, parent, tmp_path):
    train_decoder(dataset, parent, tmp_path/"legacy", epochs=1, hidden=4)
    path = tmp_path/"legacy"/"decoder.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["metadata"].pop("hidden_layers")
    torch.save(checkpoint, path)
    model, _ = load_model(parent)
    decoder, metadata = load_decoder(path, parent, model)
    assert metadata["hidden"] == 4
    assert decoder.hidden == (4,)


def test_reject_all_evaluation_separates_coverage_from_fallback_accuracy(dataset, parent, tmp_path):
    train_decoder(dataset, parent, tmp_path/"decoder", epochs=1, hidden=(8, 4), threshold=0)
    report = evaluate_model(dataset, parent, tmp_path/"evaluation",
                            decoder_path=tmp_path/"decoder"/"decoder.pt")
    assert report["coverage"]["accepted_scenarios"] == 0
    assert report["coverage"]["neural_coverage"] == 0
    assert report["accepted_accuracy"]["generator_accuracy_lt_1mw"] is None
    assert report["accepted_physics"]["feasible_rate"] is None
    assert report["hybrid_accuracy"]["generator_accuracy_lt_1mw"] == 1
    assert report["fallback_accuracy"]["joint_accuracy_lt_1mw"] == 1
    assert report["fallback_rate"] == 1
    assert report["hybrid_physics"]["feasible_rate"] == 1
    assert report["selection_curve"][0]["accepted_accuracy"]["joint_accuracy_lt_1mw"] is None
    assert report["selection_curve"][0]["accepted_physics"]["feasible_rate"] is None
    assert report["hybrid_all_requested_accuracy"]["joint_accuracy_denominator"] == 3


def test_pilot_shared_initializer_resume_and_integrity(dataset, tmp_path, monkeypatch):
    config = PilotConfig(pretrain_epochs=1, fine_tune_epochs=1, hidden=(8, 8, 8))
    before = artifact_hash(dataset/"dataset.npz")
    output = tmp_path/"pilot"
    result = run_pilot(dataset, output, config)
    assert result["completed_stages"] == 5
    assert result["held_out_evaluation_performed"] is False
    stages = result["training"]
    initializer = artifact_hash(output/stages[0]["directory"]/"model.pt")
    assert {row["initialization"]["checkpoint_sha256"] for row in stages[1:]} == {initializer}
    assert not list(output.rglob("evaluation.json"))
    assert not list(output.rglob("decoder.pt"))
    assert artifact_hash(dataset/"dataset.npz") == before

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed stages must not retrain")

    monkeypatch.setattr("powersim.pilot.train_model", forbidden)
    assert run_pilot(dataset, output, config, resume=True)["training"] == stages
    corrupted = output/stages[-1]["directory"]/"training.json"
    corrupted.write_text(corrupted.read_text()+" ")
    with pytest.raises(ValueError, match="output changed"):
        run_pilot(dataset, output, config, resume=True)
    assert not (output/".pilot.lock").exists()


def test_pilot_interrupted_attempt_is_preserved(dataset, tmp_path, monkeypatch):
    config = PilotConfig(pretrain_epochs=1, fine_tune_epochs=1, hidden=(8, 8, 8))
    output = tmp_path/"pilot"
    original = train_model

    def interrupt(dataset, directory, config, **kwargs):
        if directory.parent.name.endswith("--M2"):
            directory.mkdir(parents=True)
            (directory/"partial.txt").write_text("keep")
            raise RuntimeError("interrupted stage")
        return original(dataset, directory, config, **kwargs)

    monkeypatch.setattr("powersim.pilot.train_model", interrupt)
    with pytest.raises(RuntimeError, match="interrupted stage"):
        run_pilot(dataset, output, config)
    partial = next(output.rglob("partial.txt"))
    assert json.loads((output/"status.json").read_text())["status"] == "failed"
    monkeypatch.setattr("powersim.pilot.train_model", original)
    run_pilot(dataset, output, config, resume=True)
    assert partial.read_text() == "keep"
    state = json.loads((output/"status.json").read_text())
    assert state["stages"]["lr-0.01--M2"]["attempt"] == 2
    with pytest.raises(ValueError, match="configuration"):
        run_pilot(dataset, output, PilotConfig(
            pretrain_epochs=2, fine_tune_epochs=1, hidden=(8, 8, 8)), resume=True)


def test_cli_pilot_plan_writes_nothing(dataset, tmp_path, capsys):
    output = tmp_path/"not-created"
    assert main(["pilot", str(dataset), str(output), "--plan", "--learning-rates", "0.01,0.001"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["planned_stages"] == 10
    assert report["configuration"]["pretrain_epochs"] == 1000
    assert not output.exists()


def test_cli_train_exposes_warm_start_and_method(dataset, tmp_path, monkeypatch, capsys):
    recorded = {}

    def fake(dataset, output, config, **kwargs):
        recorded.update(configuration=asdict(config), **kwargs)
        return {}

    monkeypatch.setattr("powersim.ml.train_model", fake)
    assert main(["train", str(dataset), str(tmp_path/"out"), "--method", "M2",
                 "--initialize-from", "M1/model.pt", "--learning-rate", ".01",
                 "--epochs", "200", "--hidden", "8,8,8", "--checkpoint-selection", "last"]) == 0
    assert recorded["initialize_from"] == "M1/model.pt"
    assert recorded["configuration"]["loss_weights"] == METHOD_WEIGHTS["M2"]
    assert recorded["configuration"]["learning_rate"] == .01
    assert recorded["configuration"]["checkpoint_selection"] == "last"
    capsys.readouterr()
